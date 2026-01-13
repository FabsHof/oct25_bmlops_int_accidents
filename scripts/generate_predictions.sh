#!/bin/bash
# Script to generate random prediction data and send to API

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
API_URL="http://localhost:8000"
NUM_PREDICTIONS=${1:-10}
DELAY=${2:-1}  # delay between requests in seconds

# Get access token
echo -e "${BLUE}Getting access token...${NC}"
TOKEN=$(curl -s -X POST "$API_URL/token" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=alice&password=secret" | python3 -c "import sys, json; print(json.load(sys.stdin)['access_token'])")

if [ -z "$TOKEN" ]; then
    echo "Error: Failed to get access token"
    exit 1
fi

echo -e "${GREEN}Token obtained successfully${NC}\n"

# Function to generate random number in range
random_int() {
    local min=$1
    local max=$2
    echo $((RANDOM % (max - min + 1) + min))
}

random_float() {
    local min=$1
    local max=$2
    echo "scale=4; $min + ($(random_int 0 10000) / 10000) * ($max - $min)" | bc
}

# Function to make prediction with random values
make_prediction() {
    local iteration=$1
    
    # Generate random values for each field
    local year=$(random_int 2020 2024)
    local month=$(random_int 1 12)
    local hour=$(random_int 0 23)
    local minute=$(random_int 0 59)
    local user_category=$(random_int 1 3)
    local sex=$(random_int 1 2)
    local year_of_birth=$(random_int 1960 2005)
    local trip_purpose=$(random_int 1 5)
    local security=$(random_int 0 2)
    local luminosity=$(random_int 1 4)
    local weather=$(random_int 1 5)
    local type_of_road=$(random_int 1 5)
    local road_surface=$(random_int 1 4)
    local latitude=$(random_float 42.0 51.0)
    local longitude=$(random_float -5.0 8.0)
    local holiday=$(random_int 0 1)
    
    # Make prediction
    local response=$(curl -s -X POST "$API_URL/predict" \
      -H "Authorization: Bearer $TOKEN" \
      -H "Content-Type: application/json" \
      -d "{
        \"year\": $year,
        \"month\": $month,
        \"hour\": $hour,
        \"minute\": $minute,
        \"user_category\": $user_category,
        \"sex\": $sex,
        \"year_of_birth\": $year_of_birth,
        \"trip_purpose\": $trip_purpose,
        \"security\": $security,
        \"luminosity\": $luminosity,
        \"weather\": $weather,
        \"type_of_road\": $type_of_road,
        \"road_surface\": $road_surface,
        \"latitude\": $latitude,
        \"longitude\": $longitude,
        \"holiday\": $holiday
      }")
    
    # Extract prediction result
    local prediction=$(echo "$response" | python3 -c "import sys, json; data=json.load(sys.stdin); print(data.get('prediction_label', 'N/A'))" 2>/dev/null || echo "N/A")
    
    echo -e "${GREEN}[$(printf '%2d' $iteration)/$NUM_PREDICTIONS]${NC} Prediction: $prediction | Year: $year, Month: $month, Hour: $hour"
    
    # Delay between requests
    sleep "$DELAY"
}

# Main loop
echo -e "${BLUE}Generating $NUM_PREDICTIONS random predictions...${NC}\n"

for i in $(seq 1 $NUM_PREDICTIONS); do
    make_prediction "$i"
done

echo -e "\n${GREEN}✓ All $NUM_PREDICTIONS predictions completed!${NC}"
echo -e "${BLUE}Metrics will be available in Prometheus/Grafana in 15-30 seconds${NC}"
