import psycopg2
from psycopg2 import sql
import datetime
from datetime import datetime, timedelta

# Database configuration
conn =None
DB_NAME = "license_plate_db"
DB_USER = "postgres"
DB_PASSWORD = "m102030m"
DB_HOST = "localhost"
DB_PORT = "5432"
def drop_and_recreate_database():
    """
    Drop the existing database (if it exists) and recreate it.
    """
    #conn =None
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
        cursor.execute("""Drop TABLE IF EXISTS plates""")
        conn.commit()
        print("Table droped successfully.")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        if conn:
            cursor.close()
            conn.close()

def add_columns_to_plates_table():

    try:
        # Connect to the database
        conn = psycopg2.connect(dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD, host=DB_HOST, port=DB_PORT)
        cursor = conn.cursor()

        # Add 'sent' and 'valid' columns
        cursor.execute("""
            ALTER TABLE plates
            ADD COLUMN IF NOT EXISTS sent BOOLEAN DEFAULT FALSE,
            ADD COLUMN IF NOT EXISTS valid BOOLEAN DEFAULT FALSE;
        """)
        conn.commit()
        print("Columns 'sent' and 'valid' added successfully to the 'plates' table.")

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
    #conn =None
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
                camera_id TEXT NOT NULL,
                sent BOOLEAN DEFAULT FALSE
            )
        """)
        conn.commit()
        print("Table 'plates' created successfully")

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
    #conn =None
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
                plateimagepath TEXT NOT NULL,
                predicted_string TEXT
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
    conn =None
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
        """, (4, "testcamera", "Test Location", "rtsp://admin:123456@192.168.1.43:554/stream1"))

        conn.commit()
        print("Test camera data inserted successfully.")
    except Exception as e:
        print(f"Error inserting test camera data: {e}")
    finally:
        if conn:
            cursor.close()
            conn.close()


def add_columns_to_vehicle_table():
    """
    Add 'sent' and 'valid' columns to the 'plates' table.
    """
    try:
        # Connect to the database
        conn = psycopg2.connect(dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD, host=DB_HOST, port=DB_PORT)
        cursor = conn.cursor()

        # Add 'sent' and 'valid' columns
        cursor.execute("""
            ALTER TABLE vehicle_info
            
            ADD COLUMN IF NOT EXISTS owner_name VARCHAR(100),
            ADD COLUMN IF NOT EXISTS  organization VARCHAR(100),
            ADD COLUMN IF NOT EXISTS contact_number VARCHAR(15),
            ADD COLUMN IF NOT EXISTS  plate_image TEXT;
        """)
        conn.commit()
        print("Columns owner_name , Organizarion, contact_number and plateimage added successfully to the 'vehicle' table.")


    except Exception as e:
        print(f"Error: {e}")
    finally:
        if conn:
            cursor.close()
            conn.close()
def remove_columns():

    try:
        # Connect to the database
        conn = psycopg2.connect(dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD, host=DB_HOST, port=DB_PORT)
        cursor = conn.cursor()

        # Add 'sent' and 'valid' columns
        cursor.execute("""
            ALTER TABLE plates 
            
            DROP COLUMN owner_name ,
            DROP COLUMN organization ,
            DROP COLUMN contact_number ,
            DROP COLUMN plate_image ;
        """)
        conn.commit()
        print("Columns owner_name , Organizarion, contact_number and plateimage dropped successfully from the 'vehicle' table.")


    except Exception as e:
        print(f"Error: {e}")
    finally:
        if conn:
            cursor.close()
            conn.close()
def replace_static_with_no_slash():
    conn = psycopg2.connect(dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD, host=DB_HOST, port=DB_PORT)
    cursor = conn.cursor()

    # Update plates table - replace /static with static
    cursor.execute("""
        UPDATE plates
        SET raw_image_path = REPLACE(raw_image_path, '/static', 'static'),
            plate_cropped_image_path = REPLACE(plate_cropped_image_path, '/static', 'static')
    """)

    # Update penalties table - replace /static with static
    cursor.execute("""
        UPDATE penalties
        SET rawimagepth = REPLACE(rawimagepth, '/static', 'static'),
            plateimagepath = REPLACE(plateimagepath, '/static', 'static')
    """)

    # Commit changes and close the connection
    conn.commit()
    cursor.close()
    conn.close()

    print("Database paths updated successfully!")


def create_configuration_table():
    """
    Create the 'configuration' table to store api-key and gpsport values.
    """
    try:
        # Connect to the new database
        conn = psycopg2.connect(dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD, host=DB_HOST, port=DB_PORT)
        cursor = conn.cursor()

        # Create the 'configuration' table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS configuration (
                id SERIAL PRIMARY KEY,
                api_key TEXT NOT NULL,
                gpsport TEXT NOT NULL,
                location Text Default 'همدان'
            )
        """)
        conn.commit()
        print("Table 'configuration' created successfully with 'api_key' and 'gpsport' columns.")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        if conn:
            cursor.close()
            conn.close()

def insert_configuration():
    """
    Insert the provided api_key and gpsport into the 'configuration' table.
    """
    api_key = 'eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiIsImp0aSI6IjJjZWVjODdlMGI4Mzg5N2M4YTI3MWEzZmQ4NDQ2YmIwZTJmODE2NmExYTBlYjc1YmQ1NTNjYTVhMzBmYTQ3MjM5OWVmOTU5OGVjYzgxMjk1In0.eyJhdWQiOiIzMDA4MyIsImp0aSI6IjJjZWVjODdlMGI4Mzg5N2M4YTI3MWEzZmQ4NDQ2YmIwZTJmODE2NmExYTBlYjc1YmQ1NTNjYTVhMzBmYTQ3MjM5OWVmOTU5OGVjYzgxMjk1IiwiaWF0IjoxNzM0MzgyODYyLCJuYmYiOjE3MzQzODI4NjIsImV4cCI6MTczNjg4ODQ2Miwic3ViIjoiIiwic2NvcGVzIjpbImJhc2ljIl19.DXpzXl6tVvPqBjs-5bvRQJN5uE09XKo015Iz8nRueWmGcx7oF-TKxnLIAWi2s0jCFVbh6XXBxto3vVDsNBTaZpo5vW1qcUR6g99X_gHtfEm5UKCW6Y4nemLrXz2ihnpS1CDKvYSB-r91aoqAOYfKGvnIxFc5PWxWkhlfRxzvV0WJveIbt7O5fof9qdTJCX-ARQPYaPqNHcC8aFFpiGu0e28TsppNxce78fQnObgnXXzfqYjoAvZ1Fiqg2bVDRgDGTeuxckWPzrjKCIx0EPH5McQpFl_ukFfXqkdgE-CvOBIWBmez5BxbDukjjc0seDJlu2wP4HiLuRSn7rY9pMAi3w'
    gpsport = "COM5"
    try:
        # Connect to the database
        conn = psycopg2.connect(dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD, host=DB_HOST, port=DB_PORT)
        cursor = conn.cursor()

        # Insert api_key and gpsport into the 'configuration' table
        cursor.execute("""
            INSERT INTO configuration (api_key, gpsport)
            VALUES (%s, %s)
        """, (api_key, gpsport))
        conn.commit()
        print("api_key and gpsport inserted successfully into the 'configuration' table.")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        if conn:
            cursor.close()
            conn.close()

# Example usage:
def update_and_delete_camera_rows():
    """
    Remove rows except for ids 6, 7, 8, 9 in the 'cameras' table,
    and set their ids to 1, 2, 3, 4 respectively.
    """
    try:
        # Connect to the database
        conn = psycopg2.connect(dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD, host=DB_HOST, port=DB_PORT)
        cursor = conn.cursor()

        # Delete rows where id is not 6, 7, 8, or 9
        cursor.execute("DELETE FROM cameras WHERE id NOT IN (6, 7, 8, 9);")

        # Update the remaining rows to new id values
        cursor.execute("UPDATE cameras SET id = 1 WHERE id = 6;")
        cursor.execute("UPDATE cameras SET id = 2 WHERE id = 7;")
        cursor.execute("UPDATE cameras SET id = 3 WHERE id = 8;")
        cursor.execute("UPDATE cameras SET id = 4 WHERE id = 9;")

        # Commit the changes
        conn.commit()
        print("Rows updated and deleted successfully.")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        if conn:
            cursor.close()
            conn.close()
def add_predicted_string_to_penalties():
    """
    Add the 'predicted_string' column to the 'penalties' table and update it using the 'plates' table.
    """
    try:
        # Connect to the database
        conn = psycopg2.connect(dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD, host=DB_HOST, port=DB_PORT)
        cursor = conn.cursor()

        # Add the new column if it doesn't already exist
        cursor.execute("""
            ALTER TABLE penalties
            ADD COLUMN IF NOT EXISTS predicted_string TEXT;
        """)
        conn.commit()

        # Update the penalties table with predicted_string from plates
        cursor.execute("""
            UPDATE penalties
            SET predicted_string = plates.predicted_string
            FROM plates
            WHERE penalties.platename =CAST(plates.id AS TEXT);
        """)
        conn.commit()
        print("predicted_string added and updated successfully.")

    except Exception as e:
        print(f"Error updating penalties table: {e}")
    finally:
        if conn:
            cursor.close()
            conn.close()

# Call the function
# Call the function to update and delete rows

if __name__ == "__main__":
    # droptable()
    # create_plates_table()
    # #create_mine_info_table()
    # #remove_columns()
    # #add_columns_to_vehicle_table()
    # #create_permits_table()
    #create_configuration_table()
    #insert_configuration()
    #update_and_delete_camera_rows()
    #create_cameras_table()
    #insert_test_camera()
    add_predicted_string_to_penalties()




