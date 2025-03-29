import os
import sqlite3
from models import captioning_embedding

class CaptionDatabase:
    """Database handler for video captions."""
    
    def __init__(self, db_path='captions_test.db'):
        """Initialize the database connection.
        
        Args:
            db_path: Path to the SQLite database file
        """
        # Ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        self.create_tables()
    
    def create_tables(self):
        """Create required tables if they don't exist."""
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS captions (
                video_id TEXT,
                frame_id REAL,
                caption TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (video_id, frame_id)
            )
        ''')
        self.conn.commit()
    
    def insert_caption(self, video_id, frame_id, caption):
        """Insert a caption into the database.
        
        Args:
            video_id: ID of the video
            frame_id: Timestamp or ID of the frame
            caption: Generated caption text
        """
        self.cursor.execute(
            'INSERT INTO captions (video_id, frame_id, caption) VALUES (?, ?, ?)',
            (video_id, frame_id, caption)
        )
        self.conn.commit()
    
    def get_captions_for_video(self, video_id):
        """Get all captions for a specific video.
        
        Args:
            video_id: ID of the video
            
        Returns:
            List of (frame_id, caption) tuples
        """
        self.cursor.execute(
            'SELECT frame_id, caption FROM captions WHERE video_id = ? ORDER BY frame_id',
            (video_id,)
        )
        return self.cursor.fetchall()

    def insert_many_captions(self, captions_data):
        """Insert multiple captions at once.
        
        Args:
            captions_data: List of (video_id, frame_id, caption) tuples
        """
        self.cursor.executemany(
            'INSERT INTO captions (video_id, frame_id, caption) VALUES (?, ?, ?)',
            captions_data
        )
        self.conn.commit()
    
    def close(self):
        """Close the database connection."""
        self.conn.close()