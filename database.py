"""
Database management for MySQL and Cassandra
"""
import mysql.connector
from mysql.connector import Error
from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider
import pymongo
from pymongo import MongoClient
import pandas as pd
import json
import logging
from typing import List, Dict, Any, Optional, Union
from datetime import datetime, date
from config import config
import uuid

logger = logging.getLogger(__name__)

class MySQLManager:
    """MySQL database manager"""
    
    def __init__(self):
        self.config = config.database
        self.connection = None
        self.cursor = None
    
    def connect(self) -> bool:
        """Connect to MySQL database"""
        try:
            self.connection = mysql.connector.connect(
                host=self.config.mysql_host,
                port=self.config.mysql_port,
                user=self.config.mysql_user,
                password=self.config.mysql_password,
                database=self.config.mysql_database
            )
            self.cursor = self.connection.cursor(dictionary=True)
            logger.info("Successfully connected to MySQL database")
            return True
        except Error as e:
            logger.error(f"MySQL connection failed: {str(e)}")
            return False
    
    def disconnect(self):
        """Disconnect from MySQL database"""
        if self.cursor:
            self.cursor.close()
        if self.connection:
            self.connection.close()
        logger.info("Disconnected from MySQL database")
    
    def create_tables(self):
        """Create necessary tables for document processing"""
        tables = {
            'documents': '''
                CREATE TABLE IF NOT EXISTS documents (
                    id VARCHAR(36) PRIMARY KEY,
                    filename VARCHAR(255) NOT NULL,
                    file_type VARCHAR(50),
                    file_size INT,
                    upload_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    processing_status ENUM('pending', 'processing', 'completed', 'failed') DEFAULT 'pending',
                    document_type VARCHAR(100),
                    confidence_score FLOAT,
                    extracted_text LONGTEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
                )
            ''',
            'entities': '''
                CREATE TABLE IF NOT EXISTS entities (
                    id VARCHAR(36) PRIMARY KEY,
                    document_id VARCHAR(36),
                    entity_type VARCHAR(100),
                    entity_text TEXT,
                    confidence_score FLOAT,
                    start_position INT,
                    end_position INT,
                    extraction_method VARCHAR(50),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
                )
            ''',
            'predictions': '''
                CREATE TABLE IF NOT EXISTS predictions (
                    id VARCHAR(36) PRIMARY KEY,
                    document_id VARCHAR(36),
                    model_name VARCHAR(100),
                    prediction_type VARCHAR(100),
                    predicted_class VARCHAR(100),
                    confidence_score FLOAT,
                    prediction_data JSON,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
                )
            ''',
            'processing_logs': '''
                CREATE TABLE IF NOT EXISTS processing_logs (
                    id VARCHAR(36) PRIMARY KEY,
                    document_id VARCHAR(36),
                    processing_step VARCHAR(100),
                    status ENUM('started', 'completed', 'failed'),
                    error_message TEXT,
                    processing_time FLOAT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
                )
            '''
        }
        
        try:
            for table_name, create_sql in tables.items():
                self.cursor.execute(create_sql)
                logger.info(f"Created/verified table: {table_name}")
            
            self.connection.commit()
            return True
            
        except Error as e:
            logger.error(f"Table creation failed: {str(e)}")
            return False
    
    def insert_document(self, document_data: Dict[str, Any]) -> str:
        """Insert a new document record"""
        try:
            document_id = str(uuid.uuid4())
            
            insert_sql = '''
                INSERT INTO documents (id, filename, file_type, file_size, document_type, 
                                     confidence_score, extracted_text)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            '''
            
            values = (
                document_id,
                document_data.get('filename'),
                document_data.get('file_type'),
                document_data.get('file_size'),
                document_data.get('document_type'),
                document_data.get('confidence_score'),
                document_data.get('extracted_text')
            )
            
            self.cursor.execute(insert_sql, values)
            self.connection.commit()
            
            logger.info(f"Inserted document: {document_id}")
            return document_id
            
        except Error as e:
            logger.error(f"Document insertion failed: {str(e)}")
            return ""
    
    def insert_entities(self, document_id: str, entities: List[Dict[str, Any]]) -> bool:
        """Insert extracted entities for a document"""
        try:
            insert_sql = '''
                INSERT INTO entities (id, document_id, entity_type, entity_text, 
                                    confidence_score, start_position, end_position, extraction_method)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            '''
            
            for entity in entities:
                entity_id = str(uuid.uuid4())
                values = (
                    entity_id,
                    document_id,
                    entity.get('type', entity.get('label')),
                    entity.get('text'),
                    entity.get('confidence', 0.0),
                    entity.get('start', 0),
                    entity.get('end', 0),
                    entity.get('method', 'unknown')
                )
                
                self.cursor.execute(insert_sql, values)
            
            self.connection.commit()
            logger.info(f"Inserted {len(entities)} entities for document {document_id}")
            return True
            
        except Error as e:
            logger.error(f"Entity insertion failed: {str(e)}")
            return False
    
    def get_documents(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get documents from database"""
        try:
            select_sql = "SELECT * FROM documents ORDER BY created_at DESC LIMIT %s"
            self.cursor.execute(select_sql, (limit,))
            results = self.cursor.fetchall()
            
            return results
            
        except Error as e:
            logger.error(f"Document retrieval failed: {str(e)}")
            return []
    
    def update_processing_status(self, document_id: str, status: str) -> bool:
        """Update document processing status"""
        try:
            update_sql = "UPDATE documents SET processing_status = %s WHERE id = %s"
            self.cursor.execute(update_sql, (status, document_id))
            self.connection.commit()
            
            return True
            
        except Error as e:
            logger.error(f"Status update failed: {str(e)}")
            return False

class CassandraManager:
    """Cassandra database manager for high-volume document storage"""
    
    def __init__(self):
        self.config = config.database
        self.cluster = None
        self.session = None
        self.keyspace = self.config.cassandra_keyspace
    
    def connect(self) -> bool:
        """Connect to Cassandra cluster"""
        try:
            self.cluster = Cluster(self.config.cassandra_hosts)
            self.session = self.cluster.connect()
            logger.info("Successfully connected to Cassandra cluster")
            
            # Create keyspace if it doesn't exist
            self.create_keyspace()
            self.session.set_keyspace(self.keyspace)
            
            return True
            
        except Exception as e:
            logger.error(f"Cassandra connection failed: {str(e)}")
            return False
    
    def disconnect(self):
        """Disconnect from Cassandra"""
        if self.session:
            self.session.shutdown()
        if self.cluster:
            self.cluster.shutdown()
        logger.info("Disconnected from Cassandra cluster")
    
    def create_keyspace(self):
        """Create keyspace if it doesn't exist"""
        create_keyspace_sql = f'''
            CREATE KEYSPACE IF NOT EXISTS {self.keyspace}
            WITH REPLICATION = {{
                'class': 'SimpleStrategy',
                'replication_factor': 1
            }}
        '''
        
        self.session.execute(create_keyspace_sql)
        logger.info(f"Created/verified keyspace: {self.keyspace}")
    
    def create_tables(self):
        """Create Cassandra tables"""
        tables = {
            'document_content': '''
                CREATE TABLE IF NOT EXISTS document_content (
                    document_id UUID PRIMARY KEY,
                    filename TEXT,
                    file_type TEXT,
                    content_type TEXT,
                    raw_content BLOB,
                    extracted_text TEXT,
                    processing_metadata MAP<TEXT, TEXT>,
                    created_at TIMESTAMP,
                    updated_at TIMESTAMP
                )
            ''',
            'document_analysis': '''
                CREATE TABLE IF NOT EXISTS document_analysis (
                    document_id UUID,
                    analysis_type TEXT,
                    analysis_id UUID,
                    results TEXT,
                    confidence_score DOUBLE,
                    created_at TIMESTAMP,
                    PRIMARY KEY (document_id, analysis_type, analysis_id)
                )
            ''',
            'entity_store': '''
                CREATE TABLE IF NOT EXISTS entity_store (
                    document_id UUID,
                    entity_type TEXT,
                    entity_id UUID,
                    entity_text TEXT,
                    confidence_score DOUBLE,
                    position_data MAP<TEXT, INT>,
                    extraction_method TEXT,
                    created_at TIMESTAMP,
                    PRIMARY KEY (document_id, entity_type, entity_id)
                )
            '''
        }
        
        try:
            for table_name, create_sql in tables.items():
                self.session.execute(create_sql)
                logger.info(f"Created/verified Cassandra table: {table_name}")
            
            return True
            
        except Exception as e:
            logger.error(f"Cassandra table creation failed: {str(e)}")
            return False
    
    def store_document_content(self, document_data: Dict[str, Any]) -> str:
        """Store document content in Cassandra"""
        try:
            document_id = uuid.uuid4()
            
            insert_sql = '''
                INSERT INTO document_content (
                    document_id, filename, file_type, content_type, 
                    raw_content, extracted_text, processing_metadata, 
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            '''
            
            self.session.execute(insert_sql, (
                document_id,
                document_data.get('filename'),
                document_data.get('file_type'),
                document_data.get('content_type'),
                document_data.get('raw_content'),
                document_data.get('extracted_text'),
                document_data.get('metadata', {}),
                datetime.now(),
                datetime.now()
            ))
            
            logger.info(f"Stored document content: {document_id}")
            return str(document_id)
            
        except Exception as e:
            logger.error(f"Document storage failed: {str(e)}")
            return ""
    
    def store_analysis_results(self, document_id: str, analysis_type: str, results: Dict[str, Any]) -> bool:
        """Store analysis results"""
        try:
            analysis_id = uuid.uuid4()
            
            insert_sql = '''
                INSERT INTO document_analysis (
                    document_id, analysis_type, analysis_id, results, 
                    confidence_score, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
            '''
            
            self.session.execute(insert_sql, (
                uuid.UUID(document_id),
                analysis_type,
                analysis_id,
                json.dumps(results),
                results.get('confidence', 0.0),
                datetime.now()
            ))
            
            return True
            
        except Exception as e:
            logger.error(f"Analysis storage failed: {str(e)}")
            return False

class DatabaseManager:
    """Unified database manager"""
    
    def __init__(self):
        self.mysql = MySQLManager()
        self.cassandra = CassandraManager()
        self.mysql_connected = False
        self.cassandra_connected = False
    
    def initialize_databases(self) -> Dict[str, bool]:
        """Initialize both databases"""
        results = {}
        
        # Initialize MySQL
        try:
            self.mysql_connected = self.mysql.connect()
            if self.mysql_connected:
                self.mysql.create_tables()
                results['mysql'] = True
            else:
                results['mysql'] = False
        except Exception as e:
            logger.error(f"MySQL initialization failed: {str(e)}")
            results['mysql'] = False
        
        # Initialize Cassandra
        try:
            self.cassandra_connected = self.cassandra.connect()
            if self.cassandra_connected:
                self.cassandra.create_tables()
                results['cassandra'] = True
            else:
                results['cassandra'] = False
        except Exception as e:
            logger.error(f"Cassandra initialization failed: {str(e)}")
            results['cassandra'] = False
        
        return results
    
    def store_processed_document(self, document_data: Dict[str, Any]) -> Dict[str, str]:
        """Store processed document in both databases"""
        results = {}
        
        # Store metadata in MySQL
        if self.mysql_connected:
            try:
                mysql_id = self.mysql.insert_document(document_data)
                results['mysql_id'] = mysql_id
                
                # Store entities if available
                if 'entities' in document_data and document_data['entities']:
                    entities = self._flatten_entities(document_data['entities'])
                    self.mysql.insert_entities(mysql_id, entities)
                    
            except Exception as e:
                logger.error(f"MySQL storage failed: {str(e)}")
                results['mysql_error'] = str(e)
        
        # Store content in Cassandra
        if self.cassandra_connected:
            try:
                cassandra_id = self.cassandra.store_document_content(document_data)
                results['cassandra_id'] = cassandra_id
                
                # Store analysis results if available
                if 'predictions' in document_data:
                    self.cassandra.store_analysis_results(
                        cassandra_id, 
                        'predictions', 
                        document_data['predictions']
                    )
                    
            except Exception as e:
                logger.error(f"Cassandra storage failed: {str(e)}")
                results['cassandra_error'] = str(e)
        
        return results
    
    def _flatten_entities(self, entities_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Flatten nested entities structure for MySQL storage"""
        flattened = []
        
        def process_entities(data, prefix=''):
            if isinstance(data, dict):
                for key, value in data.items():
                    if isinstance(value, list):
                        for item in value:
                            if isinstance(item, dict):
                                item['type'] = f"{prefix}.{key}" if prefix else key
                                flattened.append(item)
                    elif isinstance(value, dict):
                        process_entities(value, f"{prefix}.{key}" if prefix else key)
        
        process_entities(entities_data)
        return flattened
    
    def get_document_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent document processing history"""
        if self.mysql_connected:
            return self.mysql.get_documents(limit)
        else:
            return []
    
    def cleanup_connections(self):
        """Clean up database connections"""
        if self.mysql_connected:
            self.mysql.disconnect()
        if self.cassandra_connected:
            self.cassandra.disconnect()
        
        logger.info("Database connections cleaned up")

# Initialize database manager
database_manager = DatabaseManager()