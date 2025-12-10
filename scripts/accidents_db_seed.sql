-- Make sure the database does not already exist, then create it and connect to it
SELECT 'CREATE DATABASE accidents_db'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'accidents_db')\gexec
\c accidents_db;
-- ============================================================
CREATE TABLE IF NOT EXISTS raw_caracteristics (
    num_acc BIGINT PRIMARY KEY,
    an INTEGER,
    mois INTEGER,
    jour INTEGER,
    hrmn INTEGER,
    lum INTEGER,
    agg INTEGER,
    int INTEGER,
    atm INTEGER,
    col INTEGER,
    com INTEGER,
    adr VARCHAR,
    gps VARCHAR,
    lat BIGINT,
    long VARCHAR,
    dep INTEGER
);
CREATE TABLE IF NOT EXISTS raw_holidays (
    id SERIAL PRIMARY KEY,
    ds DATE,
    holiday VARCHAR
);
CREATE TABLE IF NOT EXISTS raw_places (
    num_acc BIGINT PRIMARY KEY,
    catr INTEGER,
    voie INTEGER,
    v1 INTEGER,
    v2 VARCHAR,
    circ INTEGER,
    nbv INTEGER,
    pr INTEGER,
    pr1 INTEGER,
    vosp INTEGER,
    prof INTEGER,
    plan INTEGER,
    lartpc INTEGER,
    larrout INTEGER,
    surf INTEGER,
    infra INTEGER,
    situ INTEGER,
    env1 INTEGER,
    FOREIGN KEY (num_acc) REFERENCES raw_caracteristics(num_acc)
);
CREATE TABLE IF NOT EXISTS raw_users (
    user_id SERIAL PRIMARY KEY,
    num_acc BIGINT,
    place INTEGER,
    catu INTEGER,
    grav INTEGER,
    sexe INTEGER,
    trajet INTEGER,
    secu INTEGER,
    locp INTEGER,
    actp INTEGER,
    etatp INTEGER,
    an_nais INTEGER,
    num_veh VARCHAR,
    FOREIGN KEY (num_acc) REFERENCES raw_caracteristics(num_acc)
);
CREATE TABLE IF NOT EXISTS raw_vehicles (
    num_acc BIGINT,
    senc INTEGER,
    catv INTEGER,
    occutc INTEGER,
    obs INTEGER,
    obsm INTEGER,
    choc INTEGER,
    manv INTEGER,
    num_veh VARCHAR,
    PRIMARY KEY (num_acc, num_veh),
    FOREIGN KEY (num_acc) REFERENCES raw_caracteristics(num_acc)
);
CREATE TABLE IF NOT EXISTS data_ingestion_progress (
    id SERIAL PRIMARY KEY,
    table_name VARCHAR(50) UNIQUE NOT NULL,
    rows_loaded INTEGER DEFAULT 0,
    total_rows INTEGER DEFAULT 0,
    chunk_size INTEGER DEFAULT 1000,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    csv_directory VARCHAR(255),
    is_complete BOOLEAN DEFAULT FALSE
);
CREATE TABLE IF NOT EXISTS clean_data (
    record_id SERIAL PRIMARY KEY,
    num_acc BIGINT,
    raw_user_id INTEGER,
    year INTEGER,
    month INTEGER,
    hour INTEGER,
    minute INTEGER,
    user_category INTEGER,
    severity INTEGER,
    sex INTEGER,
    year_of_birth INTEGER,
    trip_purpose INTEGER,
    security INTEGER,
    luminosity INTEGER,
    weather INTEGER,
    type_of_road INTEGER,
    road_surface INTEGER,
    latitude FLOAT,
    longitude FLOAT,
    holiday BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_current BOOLEAN DEFAULT TRUE,
    valid_from TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    valid_to TIMESTAMP,
    FOREIGN KEY (num_acc) REFERENCES raw_caracteristics(num_acc)
);