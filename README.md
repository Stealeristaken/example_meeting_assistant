# Intelligent Meeting Assistant

This project is an intelligent meeting assistant developed using Azure OpenAI and LangChain. It processes natural language meeting requests and generates structured JSON output.

## 🚀 Features

- **Natural Language Processing**: Understanding meeting requests in multiple languages
- **Vector Database**: Name resolution using ChromaDB
- **Azure OpenAI Integration**: Advanced AI with GPT-4o
- **Memory Management**: Conversation history with ConversationBufferMemory
- **Ambiguity Resolution**: User clarification for multiple name matches
- **ISO 8601 Time Format**: Standard time format output
- **Email Body Generation**: Automatic professional email content

## 📋 Requirements

- Python 3.8+
- Azure OpenAI account
- Required Python packages (requirements.txt)

## 🛠️ Installation

1. **Install dependencies:**

```bash
pip install -r requirements.txt
```

2. **Set up environment variables:**

```bash
# Copy env.example to .env
cp env.example .env

# Edit .env with your Azure OpenAI credentials
# AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com/
# AZURE_OPENAI_API_KEY=your-api-key-here
```

3. **Run the application:**

```bash
python meeting_assistant_enhanced.py
```

## 🔒 Security

⚠️ **IMPORTANT**: This project no longer contains hardcoded API keys. All sensitive information is managed through environment variables.

- API keys and endpoints are stored in `.env` file
- `.env` file is excluded from Git via `.gitignore`
- Test files also use environment variables

## 🎯 Usage

### Example Meeting Requests

```
👤 YOU: Schedule a 90-minute project meeting with John and Sarah tomorrow
👤 YOU: Plan a meeting with Alex and Mike on Monday morning at 10:00
👤 YOU: Organize Q3 budget review with Lisa and Tom
👤 YOU: Quick sync with John tomorrow at 2:00 PM
```

### Output Format

```json
{
  "body": "Hello,\n\nWe have scheduled a Q3 budget review meeting.\n\nWe look forward to your participation.\n\nBest regards,",
  "endTime": "2025-07-15T10:30:00+03:00",
  "meeting_duration": 30,
  "startTime": "2025-07-15T10:00:00+03:00",
  "subject": "Q3 Budget Review",
  "user_details": [
    {
      "email_address": "lisa.smith@company.com",
      "full_name": "Lisa Smith",
      "id": 12
    },
    {
      "email_address": "tom.jones@company.com",
      "full_name": "Tom Jones",
      "id": 14
    }
  ]
}
```

## 🔧 Technical Details

### Architecture Components

1. **VectorDatabaseManager**: Name search with ChromaDB
2. **MeetingAssistantTools**: LangChain tools
3. **MeetingAssistantAgent**: Main agent class
4. **Pydantic Models**: Data validation

### Workflow

1. **Information Extraction**: Extract meeting information from user request
2. **Name Search**: Search for participants in vector database
3. **Ambiguity Resolution**: Request clarification for multiple matches
4. **Time Resolution**: Convert date/time information to ISO 8601 format
5. **Email Creation**: Generate professional email body
6. **JSON Output**: Produce final meeting JSON

### Sample Data

The system comes with 25 sample users for testing purposes.

## 🎨 Features

### Intelligent Name Resolution

- Name search using vector similarity
- Multi-language character support
- Typo tolerance
- Clarification requests for multiple matches

### Time Processing

- Natural language date/time understanding
- Business hours validation (09:00-17:00)
- Default 30-minute duration
- UTC+3 timezone

### Email Generation

- Professional email body content
- Purpose-appropriate content
- Participant information inclusion

## 🔍 Test Scenarios

### Scenario 1: Simple Meeting

```
User: "Schedule a meeting with John tomorrow"
Result: Direct JSON output
```

### Scenario 2: Ambiguous Name

```
User: "Meeting with Alex"
System: "Which Alex do you mean?"
User: "Alex Smith"
Result: JSON output
```

### Scenario 3: Missing Information

```
User: "Meeting with Sarah on Monday"
System: "What is the meeting subject?"
User: "Project status"
Result: JSON output
```

## 🚨 Error Handling

- **API Errors**: Azure OpenAI connection issues
- **Data Validation**: Format control with Pydantic
- **Ambiguity**: Request clarification from user
- **Missing Information**: Ask questions for missing fields

## 📝 License

This project is developed for educational purposes.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Submit a pull request

## 📞 Contact

For questions, please open an issue in the repository.

## 🔒 Privacy & Data Protection

This project complies with data protection regulations:

- No personal data is stored permanently
- All sample data is fictional
- User inputs are processed securely
- No personal information is logged or exposed
- Environment variables protect sensitive configuration
