
import sqlite3

# Path to your database file
db_path = r'C:\Users\Administrator\Documents\project\tb_skripsi\web_app\instance\users.db'

try:
    # Connect to the SQLite database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Execute a query to select all users
    cursor.execute("SELECT id, username, password, role FROM user")

    # Fetch all rows from the query result
    users = cursor.fetchall()

    # Check if any users were found
    if users:
        print("Daftar Pengguna:")
        # Print the header
        print(f"{ 'ID':<5} | { 'Username':<20} | { 'Password':<20} | { 'Role':<10}")
        print("-" * 60)
        # Print each user's data
        for user in users:
            print(f"{user[0]:<5} | {user[1]:<20} | {user[2]:<20} | {user[3]:<10}")
    else:
        print("Tidak ada pengguna yang ditemukan di database.")

except sqlite3.Error as e:
    print(f"Terjadi error database: {e}")

finally:
    # Close the connection
    if conn:
        conn.close()
