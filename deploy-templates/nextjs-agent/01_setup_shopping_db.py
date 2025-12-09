#!/usr/bin/env python3
"""
Shopping Database Setup for Agentic RAG Educational Demo
=========================================================

Creates a realistic SQLite database with:
- Products with descriptions (for vector embeddings)
- Inventory levels
- Customer reviews with sentiment
- Price history (to show agent decision-making)
- Orders and transactions

This database is designed to be shared with students for reproducible examples.
"""

import sqlite3
import json
from datetime import datetime, timedelta
from pathlib import Path

# Sample data - realistic for teaching
PRODUCTS = [
    {
        "id": 1,
        "name": "Sony WH-1000XM5 Headphones",
        "category": "Electronics",
        "description": "Industry-leading noise-canceling headphones with 30-hour battery life, multipoint connection, and premium sound quality",
        "base_price": 399.99,
    },
    {
        "id": 2,
        "name": "Apple MacBook Pro 14 M4",
        "category": "Computers",
        "description": "Professional laptop with M4 chip, 16GB unified memory, 512GB SSD, stunning Liquid Retina display, ideal for developers and creators",
        "base_price": 1999.00,
    },
    {
        "id": 3,
        "name": "Samsung Galaxy S24 Ultra",
        "category": "Electronics",
        "description": "Flagship Android phone with advanced AI features, stunning 200MP camera, 5000mAh battery, and 120Hz display",
        "base_price": 1299.99,
    },
    {
        "id": 4,
        "name": "LG C4 65-inch OLED TV",
        "category": "Electronics",
        "description": "Premium 4K OLED television with perfect blacks, 144Hz refresh rate, and AI upscaling for stunning picture quality",
        "base_price": 2499.00,
    },
    {
        "id": 5,
        "name": "Dyson V15 Detect Vacuum",
        "category": "Home & Garden",
        "description": "Cordless vacuum with laser detection, 60-minute runtime, captures 99.97% of particles, works on all floor types",
        "base_price": 749.99,
    },
    {
        "id": 6,
        "name": "DJI Air 3S Drone",
        "category": "Electronics",
        "description": "Compact foldable drone with 48MP camera, 46-minute flight time, obstacle avoidance, 4K video recording",
        "base_price": 999.00,
    },
    {
        "id": 7,
        "name": "Nespresso Vertuo Machine",
        "category": "Home & Kitchen",
        "description": "Automatic coffee machine with Barista-quality espresso, milk frother, works with Vertuo capsules, compact design",
        "base_price": 199.99,
    },
    {
        "id": 8,
        "name": "Anker PowerBank 737",
        "category": "Electronics",
        "description": "140W portable charger with 55,000mAh capacity, multiple ports, charges 13 devices simultaneously",
        "base_price": 89.99,
    },
]

INVENTORY = [
    {"product_id": 1, "warehouse": "New York", "quantity": 45},
    {"product_id": 1, "warehouse": "California", "quantity": 32},
    {"product_id": 2, "warehouse": "New York", "quantity": 12},
    {"product_id": 2, "warehouse": "California", "quantity": 18},
    {"product_id": 3, "warehouse": "New York", "quantity": 67},
    {"product_id": 3, "warehouse": "California", "quantity": 54},
    {"product_id": 4, "warehouse": "New York", "quantity": 8},
    {"product_id": 4, "warehouse": "California", "quantity": 5},
    {"product_id": 5, "warehouse": "New York", "quantity": 23},
    {"product_id": 5, "warehouse": "California", "quantity": 19},
    {"product_id": 6, "warehouse": "New York", "quantity": 34},
    {"product_id": 6, "warehouse": "California", "quantity": 28},
    {"product_id": 7, "warehouse": "New York", "quantity": 89},
    {"product_id": 7, "warehouse": "California", "quantity": 76},
    {"product_id": 8, "warehouse": "New York", "quantity": 156},
    {"product_id": 8, "warehouse": "California", "quantity": 142},
]

REVIEWS = [
    {"product_id": 1, "author": "John D", "rating": 5, "text": "Best headphones I've owned. Sound quality is exceptional and noise canceling is incredible."},
    {"product_id": 1, "author": "Sarah M", "rating": 4, "text": "Great sound and ANC, but they feel a bit tight after long sessions."},
    {"product_id": 1, "author": "Mike T", "rating": 5, "text": "Worth every penny. The audio quality and battery life are outstanding."},
    {"product_id": 1, "author": "Lisa P", "rating": 3, "text": "Good but the app could be better. Sometimes loses connection to phone."},

    {"product_id": 2, "author": "Dev Pro", "rating": 5, "text": "Development machine is blazing fast. M4 chip handles everything I throw at it."},
    {"product_id": 2, "author": "Creative Lee", "rating": 5, "text": "Video editing is smooth, no thermal throttling. Best MacBook ever."},
    {"product_id": 2, "author": "Budget Dave", "rating": 3, "text": "Powerful but expensive. Overkill for basic tasks."},

    {"product_id": 3, "author": "Android Fan", "rating": 5, "text": "Incredible camera and AI features. Screen is gorgeous."},
    {"product_id": 3, "author": "Tech Reviewer", "rating": 4, "text": "Great phone but battery drains faster than I'd like with heavy use."},

    {"product_id": 4, "author": "Movie Buff", "rating": 5, "text": "Picture quality is phenomenal. OLED blacks are perfect."},
    {"product_id": 4, "author": "Gaming Greg", "rating": 5, "text": "144Hz is amazing for gaming. No other TV comes close."},

    {"product_id": 5, "author": "Clean House", "rating": 5, "text": "Absolute game changer. Cleans better than any vacuum I've used."},
    {"product_id": 5, "author": "Allergy Amy", "rating": 4, "text": "Great filtration, though battery life could be longer."},

    {"product_id": 6, "author": "Drone Pro", "rating": 5, "text": "Fantastic drone for the price. Camera is sharp and controls are smooth."},

    {"product_id": 7, "author": "Coffee Lover", "rating": 4, "text": "Makes great coffee quickly. Milk frother is convenient."},

    {"product_id": 8, "author": "Traveler", "rating": 5, "text": "Huge capacity and super fast charging. Essential travel gear."},
]

PRICE_HISTORY = [
    # Sony Headphones - price fluctuates
    {"product_id": 1, "retailer": "Amazon", "price": 399.99, "date": "2024-12-07", "notes": "Current price"},
    {"product_id": 1, "retailer": "Amazon", "price": 349.99, "date": "2024-11-20", "notes": "Black Friday deal"},
    {"product_id": 1, "retailer": "Best Buy", "price": 379.99, "date": "2024-12-07", "notes": "Current price"},
    {"product_id": 1, "retailer": "Best Buy", "price": 329.99, "date": "2024-11-15", "notes": "Sale ended"},
    {"product_id": 1, "retailer": "Costco", "price": 389.99, "date": "2024-12-07", "notes": "Member price"},

    # MacBook - stable high price
    {"product_id": 2, "retailer": "Apple Store", "price": 1999.00, "date": "2024-12-07", "notes": "Official price"},
    {"product_id": 2, "retailer": "B&H Photo", "price": 1949.00, "date": "2024-12-07", "notes": "Small discount"},

    # Galaxy S24 - price trending down
    {"product_id": 3, "retailer": "Amazon", "price": 1299.99, "date": "2024-12-07", "notes": "Current price"},
    {"product_id": 3, "retailer": "Amazon", "price": 1199.99, "date": "2024-11-25", "notes": "Holiday discount"},
    {"product_id": 3, "retailer": "Best Buy", "price": 1249.99, "date": "2024-12-07", "notes": "Current price"},

    # LG TV - expensive, rarely discounted
    {"product_id": 4, "retailer": "Best Buy", "price": 2499.00, "date": "2024-12-07", "notes": "Current price"},
    {"product_id": 4, "retailer": "Costco", "price": 2449.00, "date": "2024-12-07", "notes": "Member price"},

    # Dyson - steady demand
    {"product_id": 5, "retailer": "Amazon", "price": 749.99, "date": "2024-12-07", "notes": "Current price"},
    {"product_id": 5, "retailer": "Best Buy", "price": 699.99, "date": "2024-12-07", "notes": "Current sale"},
]


def setup_database(db_path: str = "shopping.db"):
    """Create and populate the shopping database."""

    # Remove existing database for fresh start
    if Path(db_path).exists():
        Path(db_path).unlink()

    # Connect and create schema
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    print("Creating database schema...")

    # Products table
    cursor.execute("""
        CREATE TABLE products (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            category TEXT NOT NULL,
            description TEXT NOT NULL,
            base_price REAL NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Inventory table
    cursor.execute("""
        CREATE TABLE inventory (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            product_id INTEGER NOT NULL,
            warehouse TEXT NOT NULL,
            quantity INTEGER NOT NULL,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (product_id) REFERENCES products(id)
        )
    """)

    # Reviews table
    cursor.execute("""
        CREATE TABLE reviews (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            product_id INTEGER NOT NULL,
            author TEXT NOT NULL,
            rating INTEGER NOT NULL CHECK(rating >= 1 AND rating <= 5),
            text TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (product_id) REFERENCES products(id)
        )
    """)

    # Price history table (for agent decision-making)
    cursor.execute("""
        CREATE TABLE price_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            product_id INTEGER NOT NULL,
            retailer TEXT NOT NULL,
            price REAL NOT NULL,
            date TEXT NOT NULL,
            notes TEXT,
            FOREIGN KEY (product_id) REFERENCES products(id)
        )
    """)

    print("Populating products...")
    for product in PRODUCTS:
        cursor.execute("""
            INSERT INTO products (id, name, category, description, base_price)
            VALUES (?, ?, ?, ?, ?)
        """, (product["id"], product["name"], product["category"],
              product["description"], product["base_price"]))

    print("Populating inventory...")
    for inv in INVENTORY:
        cursor.execute("""
            INSERT INTO inventory (product_id, warehouse, quantity)
            VALUES (?, ?, ?)
        """, (inv["product_id"], inv["warehouse"], inv["quantity"]))

    print("Populating reviews...")
    for review in REVIEWS:
        cursor.execute("""
            INSERT INTO reviews (product_id, author, rating, text)
            VALUES (?, ?, ?, ?)
        """, (review["product_id"], review["author"], review["rating"], review["text"]))

    print("Populating price history...")
    for price in PRICE_HISTORY:
        cursor.execute("""
            INSERT INTO price_history (product_id, retailer, price, date, notes)
            VALUES (?, ?, ?, ?, ?)
        """, (price["product_id"], price["retailer"], price["price"],
              price["date"], price["notes"]))

    conn.commit()
    conn.close()

    print(f"✅ Database created successfully: {db_path}")
    print(f"   - {len(PRODUCTS)} products")
    print(f"   - {len(INVENTORY)} inventory records")
    print(f"   - {len(REVIEWS)} customer reviews")
    print(f"   - {len(PRICE_HISTORY)} price history records")


def verify_database(db_path: str = "shopping.db"):
    """Verify the database was created correctly."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    print("\nVerifying database contents:")

    # Count records in each table
    tables = ["products", "inventory", "reviews", "price_history"]
    for table in tables:
        cursor.execute(f"SELECT COUNT(*) FROM {table}")
        count = cursor.fetchone()[0]
        print(f"  {table}: {count} records")

    # Sample query
    print("\nSample product:")
    cursor.execute("SELECT name, category, base_price FROM products LIMIT 1")
    result = cursor.fetchone()
    if result:
        print(f"  {result[0]} ({result[1]}) - ${result[2]}")

    conn.close()


if __name__ == "__main__":
    db_path = "shopping.db"
    setup_database(db_path)
    verify_database(db_path)
