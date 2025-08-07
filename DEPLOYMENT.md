# AstroAssistance Deployment Guide

This guide provides instructions for deploying and running the AstroAssistance system, a self-learning AI productivity assistant.

## System Requirements

- Python 3.8+ (Python 3.13 compatible)
- PostgreSQL database (for production deployment)
- Linux, macOS, or Windows operating system

## Quick Start

### 1. Setup and Deployment

The deployment script handles environment setup, dependency installation, and service startup:

```bash
# Clone the repository (if not already done)
git clone https://github.com/yourusername/AstroAssistance.git
cd AstroAssistance

# Make the deployment script executable
chmod +x deploy.py

# Run the deployment script with setup-only flag to prepare the environment
./deploy.py --setup-only --generate-data

# Start the API server
./deploy.py
```

### 2. Testing the API

Use the test client to interact with the API:

```bash
# Make the test client executable
chmod +x test_client.py

# Run the test client to view tasks and recommendations
./test_client.py

# Create a new task
./test_client.py --create-task --title "New Task" --description "Task description" --priority HIGH --category WORK --duration 60
```

### 3. API Documentation

Access the API documentation at: http://localhost:8000/docs

## Production Deployment

For production deployment, consider the following:

1. **Database Configuration**:
   - Set up a PostgreSQL database
   - Configure database connection in `config/config.yaml`

2. **Process Management**:
   - Use a process manager like supervisord or systemd to manage the API server process
   - Example systemd service file:

   ```ini
   [Unit]
   Description=AstroAssistance API Server
   After=network.target

   [Service]
   User=appuser
   WorkingDirectory=/path/to/AstroAssistance
   ExecStart=/path/to/AstroAssistance/venv/bin/python /path/to/AstroAssistance/minimal_api.py
   Restart=on-failure

   [Install]
   WantedBy=multi-user.target
   ```

3. **Security**:
   - Set a secure JWT secret key in environment variables
   - Configure CORS settings for production
   - Use HTTPS with a proper SSL certificate

4. **Monitoring**:
   - Set up monitoring with Prometheus and Grafana
   - Configure logging to a centralized log management system

## Troubleshooting

### Common Issues

1. **Dependency Installation Failures**:
   - If you encounter issues with specific packages, try installing them individually
   - For C extension issues, ensure you have the appropriate development tools installed

2. **Database Connection Issues**:
   - Verify PostgreSQL is running
   - Check connection string in configuration
   - Ensure database user has appropriate permissions

3. **API Server Not Starting**:
   - Check logs for error messages
   - Verify port 8000 is not in use by another application

### Getting Help

For additional assistance, please:
- Check the project documentation in the `docs` directory
- Open an issue on the GitHub repository
- Contact the development team at support@astroassistance.example.com

## Advanced Configuration

For advanced configuration options, refer to the `config/config.yaml` file. Key configuration sections include:

- `api`: API server settings
- `database`: Database connection settings
- `models`: Machine learning model configurations
- `training`: Model training parameters
- `features`: Feature toggles for different capabilities

## License

This project is licensed under the MIT License - see the LICENSE file for details.