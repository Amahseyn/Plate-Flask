import psycopg2
from psycopg2 import sql
import datetime
from datetime import datetime, timedelta

# Database configuration

DB_NAME = "license_plate_db"
DB_USER = "postgres"
DB_PASSWORD = "m102030m"
DB_HOST = "localhost"
DB_PORT = "5432"
def drop_and_recreate_database():
    """
    Drop the existing database (if it exists) and recreate it.
    """
    try:
        # Connect to the default 'postgres' database as a superuser
        conn = psycopg2.connect(dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD, host=DB_HOST, port=DB_PORT)
        conn.autocommit = True
        cursor = conn.cursor()

        # Drop the existing database
        cursor.execute(sql.SQL("DROP DATABASE IF EXISTS {db_name}").format(db_name=sql.Identifier(DB_NAME)))
        print(f"Database '{DB_NAME}' dropped successfully (if it existed).")

        # Create a fresh database
        cursor.execute(sql.SQL("CREATE DATABASE {db_name}").format(db_name=sql.Identifier(DB_NAME)))
        print(f"Database '{DB_NAME}' created successfully.")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        if conn:
            cursor.close()
            conn.close()
def droptable():
    """
    Create the 'cameras' table in the newly created database.
    """
    try:
        # Connect to the new database
        conn = psycopg2.connect(dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD, host=DB_HOST, port=DB_PORT)
        cursor = conn.cursor()

        # Create the 'cameras' table
        cursor.execute("""Drop TABLE IF EXISTS penalties""")
        conn.commit()
        print("Table droped successfully.")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        if conn:
            cursor.close()
            conn.close()
def create_plates_table():
    """
    Create the 'plates' table in the newly created database.
    """
    try:
        # Connect to the new database
        conn = psycopg2.connect(dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD, host=DB_HOST, port=DB_PORT)
        cursor = conn.cursor()

        # Create the 'plates' table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS plates (
                id SERIAL PRIMARY KEY,
                date TEXT NOT NULL,
                raw_image_path TEXT NOT NULL,
                plate_cropped_image_path TEXT NOT NULL,
                predicted_string TEXT NOT NULL,
                camera_id TEXT NOT NULL
            )
        """)
        conn.commit()
        print("Table 'plates' created successfully.")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        if conn:
            cursor.close()
            conn.close()
def create_penalties_table():
    """
    Create the 'penalties' table in the newly created database.
    """
    try:
        # Connect to the new database
        conn = psycopg2.connect(dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD, host=DB_HOST, port=DB_PORT)
        cursor = conn.cursor()

        # Create the 'penalties' table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS penalties (
                id SERIAL PRIMARY KEY,
                platename TEXT NOT NULL,
                penaltytype TEXT NOT NULL,
                location TEXT NOT NULL,
                datetime Text NOT NULL,
                rawimagepth TEXT NOT NULL,
                plateimagepath TEXT NOT NULL
            )
        """)
        conn.commit()
        print("Table 'penalties' created successfully.")

    except Exception as e:
        print(f"Error creating 'penalties' table: {e}")
    finally:
        if conn:
            cursor.close()
            conn.close()
def create_cameras_table():
    """
    Create the 'cameras' table in the newly created database.
    """
    try:
        # Connect to the new database
        conn = psycopg2.connect(dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD, host=DB_HOST, port=DB_PORT)
        cursor = conn.cursor()

        # Create the 'cameras' table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS cameras (
                id SERIAL PRIMARY KEY,
                cameraname TEXT NOT NULL,
                cameralocation TEXT NOT NULL,
                cameralink TEXT NOT NULL
            )
        """)
        conn.commit()
        print("Table 'cameras' created successfully.")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        if conn:
            cursor.close()
            conn.close()
def insert_test_camera():
    """
    Insert test data into the 'cameras' table.
    """
    try:
        conn = psycopg2.connect(dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD, host=DB_HOST, port=DB_PORT)
        cursor = conn.cursor()

        # Insert the test camera data
        cursor.execute("""
            INSERT INTO cameras (id, cameraname, cameralocation, cameralink)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (id) DO NOTHING
        """, (2, "testcamera", "Test Location", "a09.mp4"))

        conn.commit()
        print("Test camera data inserted successfully.")
    except Exception as e:
        print(f"Error inserting test camera data: {e}")
    finally:
        if conn:
            cursor.close()
            conn.close()
def insert_test_penalty():
    """
    Insert test data into the 'penalties' table.
    """
    try:
        conn = psycopg2.connect(dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD, host=DB_HOST, port=DB_PORT)
        cursor = conn.cursor()
        current_time = datetime.now()
        datetime_value = str(current_time.strftime("%Y-%m-%d-%H-%M-%S"))
        print(datetime_value)
        # Insert test penalty data
        cursor.execute("""
            INSERT INTO penalties (platename, penaltytype, location, datetime, rawimagepth, plateimagepath)
            VALUES (%s, %s, %s, %s, %s,%s)
        """, ("AK48", "test", "location5", "2024-12-05-17-54-54", "/images/raw5.jpg","images/plt.png"))

        conn.commit()
        print("Test penalty data inserted successfully.")
    except Exception as e:
        print(f"Error inserting test penalty data: {e}")
    finally:
        if conn:
            cursor.close()
            conn.close()

if __name__ == "__main__":
    drop_and_recreate_database()
    create_plates_table()
    create_cameras_table()
    create_penalties_table()