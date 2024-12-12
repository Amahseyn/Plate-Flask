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
        cursor.execute("""Drop TABLE IF EXISTS vehicles""")
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
def create_permits_table():
    """
    Create a combined table 'vehicle_permit' with auto-incrementing vehicle_id starting from 0.
    """
    try:
        # Connect to the database
        conn = psycopg2.connect(dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD, host=DB_HOST, port=DB_PORT)
        cursor = conn.cursor()

        # Create the 'vehicle_info' table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS vehicle_info (
                vehicle_id SERIAL PRIMARY KEY,  -- Auto-incrementing ID
                license_plate VARCHAR(20) UNIQUE NOT NULL
            )
        """)

        # Create the 'vehicle_permit' table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS vehicle_permit (
                permit_id SERIAL PRIMARY KEY,
                vehicle_id INT NOT NULL REFERENCES vehicle_info(vehicle_id),
                mine_id INT NOT NULL,
                start_date TEXT NOT NULL,
                end_date TEXT NOT NULL
            )
        """)
        conn.commit()
        print("Tables 'vehicle_info' and 'vehicle_permit' created successfully.")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        if conn:
            cursor.close()
            conn.close()
def create_mine_info_table():
    """
    Create the 'mine_info' table to store information about mines.
    """
    try:
        # Connect to the database
        conn = psycopg2.connect(dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD, host=DB_HOST, port=DB_PORT)
        cursor = conn.cursor()

        # Create the 'mine_info' table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS mine_info (
                mine_id SERIAL PRIMARY KEY,  -- Auto-incrementing ID for each mine
                mine_name VARCHAR(100) NOT NULL,  -- Name of the mine
                location VARCHAR(100),  -- Location of the mine
                owner_name VARCHAR(100),  -- Owner's name of the mine
                contact_number VARCHAR(15)  -- Contact number for the mine
            )
        """)

        conn.commit()
        print("Table 'mine_info' created successfully.")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        if conn:
            cursor.close()
            conn.close()



# Example usage
if __name__ == "__main__":
    create_mine_info_table()

