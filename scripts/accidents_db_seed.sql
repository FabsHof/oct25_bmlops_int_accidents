CREATE DATABASE accidents_db;
\c accidents_db;
CREATE TABLE raw_caracteristics (
    num_acc INTEGER PRIMARY KEY, --would SERIAL be better? the num_acc are already defined
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
    lat INTEGER,
    long INTEGER,
    dep INTEGER
);
CREATE TABLE raw_holidays (
    ds DATE PRIMARY KEY,
    holiday VARCHAR
);
CREATE TABLE raw_places (
    num_acc INTEGER PRIMARY KEY,
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
CREATE TABLE raw_users (
    user_id SERIAL PRIMARY KEY,
    num_acc INTEGER,
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
CREATE TABLE raw_vehicles (
    num_acc INTEGER,
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
CREATE TABLE preprocessed_data (
    user_id SERIAL PRIMARY KEY,
    num_acc INTEGER,
    year_ INTEGER,
    moy_cos FLOAT,
    moy_sin FLOAT,
    dow_cos FLOAT,
    dow_sin FLOAT,
    hod_cos FLOAT,
    hod_sin FLOAT,
    catu INTEGER,
    grav INTEGER,
    sex INTEGER,
    birthyear INTEGER,
    purpose INTEGER,
    securiy INTEGER,
    luminosity INTEGER,
    weather INTEGER,
    road_type INTEGER,
    road_surface INTEGER,
    latitude INTEGER,
    longitude INTEGER,
    holiday BOOLEAN,
    FOREIGN KEY (user_id) REFERENCES raw_users(user_id),
    FOREIGN KEY (num_acc) REFERENCES raw_caracteristics(num_acc)
);