# Python engineer technical assessment

## Overview
Design and implement a Scalable Document Analysis System (pipeline) that leverages AI to process, analyze, and extract insights from large collections of documents while maintaining high performance and reliability.

## Project Requirements

### Core System Requirements

1. Create a document processing service that can:
   - Handle multiple document formats (PDF, DOCX, JSON)
   - Process documents in parallel
   - Extract text and maintain document structure
   - Generate document embeddings
   - Perform semantic search across documents
   - Classify documents into categories
   - Extract key information using custom NER models

### Technical Requirements

#### Python Implementation
- Implement the solution using Python 
- Create modular and extensible code
- Include proper error handling and logging
- Implement unit tests and integration tests

#### AI/ML Components
- Implement document embedding generation using a model of your choice (explain your selection)
- Create a custom NER model for information extraction
- Implement a document classification system
- Design a semantic search functionality

#### Infrastructure
- Containerize the solution using Docker
- Create a docker-compose setup for local development
- Design the system to be cloud-ready (No need to deploy the solution, just provide the architecture diagram)

## Evaluation Criteria

Your solution will be evaluated based on:

1. **Code Quality**
   - Clean, readable, and maintainable code
   - Proper use of design patterns
   - Error handling and logging
   - Testing coverage
   - Documentation quality

2. **System Design**
   - Architecture scalability
   - Component isolation
   - Resource efficiency
   - Error resilience
   - Monitoring capabilities

3. **AI Implementation**
   - Model selection justification
   - Implementation efficiency
   - Accuracy and performance
   - Training and evaluation methodology

4. **Innovation**
   - Creative solutions to problems
   - Unique features or improvements
   - Performance optimizations

## Deliverables

1. **Source Code**
   - Complete source code with documentation (github repository, you can fork this or branch)
   - Setup instructions

2. **Documentation**
   - System architecture diagram
   - Model selection justification
   - Performance analysis
   - Scaling considerations

3. **Docker Configuration**
   - Environment configuration
   - Build and run instructions

4. **Presentation**
   - Brief presentation explaining:
     - Architecture decisions
     - Model selections
     - Scaling strategy
     - Future improvements

## Bonus Points

- Implementation of A/B testing for model deployment
- Advanced monitoring and alerting setup
- Performance optimization techniques
- Novel approaches to document processing
- Advanced caching strategies

## Time Allocation

- Candidates should spend 3-5 days on this project
- Focus on demonstrating knowledge rather than completing every feature
- Prioritize core functionality and code quality
- Document any assumptions and future improvements

## Notes

- You can use any open-source libraries and models
- Explain your choice of technologies and frameworks
- Include any assumptions made during implementation
- Document known limitations and potential improvements
- Focus on demonstrating your problem-solving approach
- Create a clear path for scaling the solution

### Very important: you can request information about the test and communicate via email to otorres@auraresearch.ai or oscar@auraresearch.ai to resolve doubts and discuss technical proposals. Communication and way of working will be evaluated. Not all points are indispensable, but a good argumentation of the architecture/pipeline is required.
