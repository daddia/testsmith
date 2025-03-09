-- Complex SQL script with more sophisticated procedures and functions

-- Create database schema
CREATE TABLE IF NOT EXISTS products (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    description TEXT,
    price DECIMAL(10, 2) NOT NULL,
    stock_quantity INTEGER NOT NULL DEFAULT 0,
    category VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS customers (
    id SERIAL PRIMARY KEY,
    first_name VARCHAR(50) NOT NULL,
    last_name VARCHAR(50) NOT NULL,
    email VARCHAR(100) NOT NULL UNIQUE,
    phone VARCHAR(20),
    address TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS orders (
    id SERIAL PRIMARY KEY,
    customer_id INTEGER NOT NULL REFERENCES customers(id),
    order_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    total_amount DECIMAL(12, 2) NOT NULL,
    status VARCHAR(20) NOT NULL DEFAULT 'pending',
    shipping_address TEXT,
    tracking_number VARCHAR(50),
    CONSTRAINT valid_status CHECK (status IN ('pending', 'processing', 'shipped', 'delivered', 'cancelled'))
);

CREATE TABLE IF NOT EXISTS order_items (
    id SERIAL PRIMARY KEY,
    order_id INTEGER NOT NULL REFERENCES orders(id),
    product_id INTEGER NOT NULL REFERENCES products(id),
    quantity INTEGER NOT NULL,
    unit_price DECIMAL(10, 2) NOT NULL,
    subtotal DECIMAL(12, 2) NOT NULL
);

-- Create complex functions and procedures

-- Function to calculate product's average rating
CREATE OR REPLACE FUNCTION get_product_rating(p_product_id INTEGER)
RETURNS DECIMAL(3, 2) AS $$
DECLARE
    avg_rating DECIMAL(3, 2);
BEGIN
    -- In a real scenario, this would query a reviews table
    -- This is a simplified example
    SELECT CASE 
        WHEN p_product_id % 5 = 0 THEN 5.0
        WHEN p_product_id % 5 = 1 THEN 4.5
        WHEN p_product_id % 5 = 2 THEN 4.0
        WHEN p_product_id % 5 = 3 THEN 3.5
        ELSE 4.2
    END INTO avg_rating;
    
    RETURN avg_rating;
END;
$$ LANGUAGE plpgsql;

-- Function to check if a product is in stock
CREATE OR REPLACE FUNCTION is_product_in_stock(p_product_id INTEGER, p_quantity INTEGER DEFAULT 1)
RETURNS BOOLEAN AS $$
DECLARE
    available_quantity INTEGER;
BEGIN
    SELECT stock_quantity 
    INTO available_quantity 
    FROM products 
    WHERE id = p_product_id;
    
    IF available_quantity IS NULL THEN
        RAISE EXCEPTION 'Product with ID % not found', p_product_id;
    END IF;
    
    RETURN available_quantity >= p_quantity;
END;
$$ LANGUAGE plpgsql;

-- Function to update product stock
CREATE OR REPLACE FUNCTION update_product_stock(p_product_id INTEGER, p_quantity_change INTEGER)
RETURNS INTEGER AS $$
DECLARE
    new_quantity INTEGER;
BEGIN
    UPDATE products
    SET 
        stock_quantity = stock_quantity + p_quantity_change,
        updated_at = CURRENT_TIMESTAMP
    WHERE id = p_product_id
    RETURNING stock_quantity INTO new_quantity;
    
    IF new_quantity IS NULL THEN
        RAISE EXCEPTION 'Product with ID % not found', p_product_id;
    END IF;
    
    IF new_quantity < 0 THEN
        RAISE EXCEPTION 'Cannot reduce stock below zero (Product ID: %)', p_product_id;
    END IF;
    
    RETURN new_quantity;
END;
$$ LANGUAGE plpgsql;

-- Function to calculate order total
CREATE OR REPLACE FUNCTION calculate_order_total(p_order_id INTEGER)
RETURNS DECIMAL(12, 2) AS $$
DECLARE
    total DECIMAL(12, 2);
BEGIN
    SELECT SUM(subtotal)
    INTO total
    FROM order_items
    WHERE order_id = p_order_id;
    
    IF total IS NULL THEN
        -- No items in order or order doesn't exist
        RETURN 0;
    END IF;
    
    RETURN total;
END;
$$ LANGUAGE plpgsql;

-- Procedure to create a new order
CREATE OR REPLACE PROCEDURE create_order(
    p_customer_id INTEGER,
    p_shipping_address TEXT,
    OUT p_order_id INTEGER
)
LANGUAGE plpgsql AS $$
BEGIN
    -- Check if customer exists
    IF NOT EXISTS (SELECT 1 FROM customers WHERE id = p_customer_id) THEN
        RAISE EXCEPTION 'Customer with ID % not found', p_customer_id;
    END IF;
    
    -- Create the order
    INSERT INTO orders (customer_id, total_amount, shipping_address)
    VALUES (p_customer_id, 0, p_shipping_address)
    RETURNING id INTO p_order_id;
END;
$$;

-- Procedure to add item to an order
CREATE OR REPLACE PROCEDURE add_order_item(
    p_order_id INTEGER,
    p_product_id INTEGER,
    p_quantity INTEGER
)
LANGUAGE plpgsql AS $$
DECLARE
    v_price DECIMAL(10, 2);
    v_subtotal DECIMAL(12, 2);
    v_order_status VARCHAR(20);
BEGIN
    -- Check if order exists and is in a valid state for modification
    SELECT status INTO v_order_status FROM orders WHERE id = p_order_id;
    
    IF v_order_status IS NULL THEN
        RAISE EXCEPTION 'Order with ID % not found', p_order_id;
    END IF;
    
    IF v_order_status NOT IN ('pending', 'processing') THEN
        RAISE EXCEPTION 'Cannot modify order in % status', v_order_status;
    END IF;
    
    -- Check if product exists and get its price
    SELECT price INTO v_price FROM products WHERE id = p_product_id;
    
    IF v_price IS NULL THEN
        RAISE EXCEPTION 'Product with ID % not found', p_product_id;
    END IF;
    
    -- Check if product is in stock
    IF NOT is_product_in_stock(p_product_id, p_quantity) THEN
        RAISE EXCEPTION 'Insufficient stock for product ID %', p_product_id;
    END IF;
    
    -- Calculate subtotal
    v_subtotal := v_price * p_quantity;
    
    -- Add order item
    INSERT INTO order_items (order_id, product_id, quantity, unit_price, subtotal)
    VALUES (p_order_id, p_product_id, p_quantity, v_price, v_subtotal);
    
    -- Update product stock
    PERFORM update_product_stock(p_product_id, -p_quantity);
    
    -- Update order total
    UPDATE orders
    SET total_amount = calculate_order_total(p_order_id)
    WHERE id = p_order_id;
END;
$$;

-- Function to get customer order history with pagination
CREATE OR REPLACE FUNCTION get_customer_orders(
    p_customer_id INTEGER,
    p_page INTEGER DEFAULT 1,
    p_page_size INTEGER DEFAULT 10
)
RETURNS TABLE (
    order_id INTEGER,
    order_date TIMESTAMP,
    total_amount DECIMAL(12, 2),
    status VARCHAR(20),
    item_count BIGINT
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        o.id AS order_id,
        o.order_date,
        o.total_amount,
        o.status,
        COUNT(oi.id) AS item_count
    FROM orders o
    LEFT JOIN order_items oi ON o.id = oi.order_id
    WHERE o.customer_id = p_customer_id
    GROUP BY o.id, o.order_date, o.total_amount, o.status
    ORDER BY o.order_date DESC
    LIMIT p_page_size
    OFFSET (p_page - 1) * p_page_size;
END;
$$ LANGUAGE plpgsql;