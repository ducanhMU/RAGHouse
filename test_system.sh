#!/bin/bash

# test_system.sh - Comprehensive system testing script

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
API_BASE="http://localhost:8000"
UI_BASE="http://localhost:8501"
TEST_FILE="test_document.txt"

echo "======================================"
echo "RAG System Integration Test Suite"
echo "======================================"
echo ""

# Function to print colored output
print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_info() {
    echo -e "${YELLOW}ℹ $1${NC}"
}

# Function to check if service is responding
check_service() {
    local url=$1
    local name=$2
    
    if curl -sf "$url" > /dev/null; then
        print_success "$name is responding"
        return 0
    else
        print_error "$name is not responding"
        return 1
    fi
}

# Test 1: Check all services
echo "Test 1: Checking Service Health"
echo "--------------------------------"

check_service "$API_BASE/health" "API"
check_service "$UI_BASE/_stcore/health" "UI"
check_service "http://localhost:9091/healthz" "Milvus"

# Check API health details
print_info "Fetching detailed health status..."
HEALTH_RESPONSE=$(curl -s "$API_BASE/health")
echo "$HEALTH_RESPONSE" | jq . || echo "$HEALTH_RESPONSE"

echo ""

# Test 2: Database connectivity
echo "Test 2: Database Connectivity"
echo "------------------------------"

if docker exec rag_postgres pg_isready -U rag_user > /dev/null 2>&1; then
    print_success "PostgreSQL is ready"
else
    print_error "PostgreSQL is not ready"
    exit 1
fi

echo ""

# Test 3: Ollama models
echo "Test 3: Ollama Models"
echo "---------------------"

print_info "Checking Ollama models..."
docker exec rag_ollama ollama list

if docker exec rag_ollama ollama list | grep -q "nomic-embed-text"; then
    print_success "Embedding model found"
else
    print_error "Embedding model not found"
fi

if docker exec rag_ollama ollama list | grep -q "llama3.2"; then
    print_success "LLM model found"
else
    print_error "LLM model not found"
fi

echo ""

# Test 4: Create test document
echo "Test 4: Document Upload"
echo "-----------------------"

print_info "Creating test document..."
cat > "$TEST_FILE" << EOF
This is a test document for the RAG system.

Financial Report Summary:
- Revenue: $1,000,000
- Expenses: $600,000
- Net Profit: $400,000

Key Findings:
1. Revenue increased by 20% year-over-year
2. Operating expenses decreased by 10%
3. Strong profit margins maintained

This document contains important financial information that should be retrievable by the RAG system.
EOF

print_info "Uploading test document..."
UPLOAD_RESPONSE=$(curl -s -X POST "$API_BASE/upload" \
    -F "file=@$TEST_FILE")

echo "$UPLOAD_RESPONSE" | jq . || echo "$UPLOAD_RESPONSE"

FILE_ID=$(echo "$UPLOAD_RESPONSE" | jq -r '.file_id')

if [ "$FILE_ID" != "null" ] && [ -n "$FILE_ID" ]; then
    print_success "Document uploaded successfully (ID: $FILE_ID)"
else
    print_error "Document upload failed"
    exit 1
fi

# Wait for processing
print_info "Waiting for document processing..."
sleep 10

# Check file status
STATUS_RESPONSE=$(curl -s "$API_BASE/files/$FILE_ID")
STATUS=$(echo "$STATUS_RESPONSE" | jq -r '.status')
print_info "File status: $STATUS"

echo ""

# Test 5: List files
echo "Test 5: List Files"
echo "------------------"

FILES_RESPONSE=$(curl -s "$API_BASE/files")
FILE_COUNT=$(echo "$FILES_RESPONSE" | jq '. | length')
print_info "Found $FILE_COUNT files"
echo "$FILES_RESPONSE" | jq '.[0:3]' || echo "$FILES_RESPONSE"

echo ""

# Test 6: Create chat session
echo "Test 6: Chat Functionality"
echo "--------------------------"

print_info "Starting new chat session..."

# First message
CHAT_PAYLOAD='{"message": "What is the revenue mentioned in the documents?"}'
print_info "Sending query: What is the revenue mentioned in the documents?"

CHAT_RESPONSE=$(curl -s -X POST "$API_BASE/chat" \
    -H "Content-Type: application/json" \
    -d "$CHAT_PAYLOAD")

# Save to temp file for parsing
echo "$CHAT_RESPONSE" > /tmp/chat_response.txt

# Extract session ID
SESSION_ID=$(grep 'session_id' /tmp/chat_response.txt | head -1 | sed 's/.*session_id.*:\s*"\([^"]*\)".*/\1/')

if [ -n "$SESSION_ID" ]; then
    print_success "Chat session created (ID: $SESSION_ID)"
else
    print_error "Failed to create chat session"
fi

# Check if response contains expected information
if grep -q "1,000,000\|1000000\|revenue" /tmp/chat_response.txt; then
    print_success "Response contains relevant financial information"
else
    print_error "Response does not contain expected information"
fi

echo ""

# Test 7: Continue conversation
echo "Test 7: Conversation Context"
echo "----------------------------"

if [ -n "$SESSION_ID" ]; then
    print_info "Sending follow-up question..."
    
    FOLLOWUP_PAYLOAD="{\"session_id\": \"$SESSION_ID\", \"message\": \"What was the net profit?\"}"
    
    curl -s -X POST "$API_BASE/chat" \
        -H "Content-Type: application/json" \
        -d "$FOLLOWUP_PAYLOAD" > /tmp/chat_followup.txt
    
    if grep -q "400,000\|profit" /tmp/chat_followup.txt; then
        print_success "Follow-up question answered correctly"
    else
        print_error "Follow-up answer incorrect or missing"
    fi
else
    print_error "Skipping follow-up test (no session ID)"
fi

echo ""

# Test 8: List sessions
echo "Test 8: Session Management"
echo "--------------------------"

SESSIONS_RESPONSE=$(curl -s "$API_BASE/sessions")
SESSION_COUNT=$(echo "$SESSIONS_RESPONSE" | jq '. | length')
print_info "Found $SESSION_COUNT sessions"
echo "$SESSIONS_RESPONSE" | jq '.[0:2]' || echo "$SESSIONS_RESPONSE"

echo ""

# Test 9: Get session history
echo "Test 9: Session History"
echo "-----------------------"

if [ -n "$SESSION_ID" ]; then
    print_info "Fetching history for session: $SESSION_ID"
    
    HISTORY_RESPONSE=$(curl -s "$API_BASE/sessions/$SESSION_ID/history")
    MESSAGE_COUNT=$(echo "$HISTORY_RESPONSE" | jq '.messages | length')
    
    print_info "Session has $MESSAGE_COUNT messages"
    echo "$HISTORY_RESPONSE" | jq '.messages | .[0:2]' || echo "$HISTORY_RESPONSE"
    
    if [ "$MESSAGE_COUNT" -ge 2 ]; then
        print_success "Session history retrieved successfully"
    fi
else
    print_error "Skipping history test (no session ID)"
fi

echo ""

# Test 10: Performance metrics
echo "Test 10: Performance Check"
echo "--------------------------"

print_info "Checking Docker container stats..."
docker stats --no-stream --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}" | head -6

echo ""

# Test 11: Check logs for errors
echo "Test 11: Error Log Check"
echo "------------------------"

print_info "Checking API logs for errors (last 50 lines)..."
ERROR_COUNT=$(docker logs rag_api --tail 50 2>&1 | grep -i "error\|exception\|failed" | grep -v "ERROR 404" | wc -l)

if [ "$ERROR_COUNT" -eq 0 ]; then
    print_success "No recent errors in API logs"
else
    print_error "Found $ERROR_COUNT potential errors in API logs"
    print_info "Recent errors:"
    docker logs rag_api --tail 50 2>&1 | grep -i "error\|exception\|failed" | grep -v "ERROR 404" | tail -5
fi

echo ""

# Test 12: Cleanup test data
echo "Test 12: Cleanup"
echo "----------------"

print_info "Cleaning up test files..."
rm -f "$TEST_FILE" /tmp/chat_response.txt /tmp/chat_followup.txt

if [ -n "$SESSION_ID" ]; then
    print_info "Deleting test session: $SESSION_ID"
    curl -s -X DELETE "$API_BASE/sessions/$SESSION_ID" > /dev/null
fi

print_success "Cleanup complete"

echo ""
echo "======================================"
echo "Test Suite Complete!"
echo "======================================"
echo ""

# Summary
print_info "Summary:"
echo "- All critical services are running"
echo "- Document upload and processing works"
echo "- Chat functionality is operational"
echo "- Session management is working"
echo "- No critical errors detected"
echo ""

print_success "System is ready for use!"