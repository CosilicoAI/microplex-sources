# Microdata Authentication

## Public Datasets (no auth required)
- CPS ASEC - Census Bureau public download
- ACS PUMS - Census Bureau public download

## Restricted Datasets (auth needed)

### IRS Public Use File (PUF)
- Source: IRS Statistics of Income
- Access: Requires approved research project
- TODO: Add credential management for authenticated downloads

### UK Family Resources Survey (FRS)
- Source: UK Data Service
- Access: Requires UK Data Service registration
- TODO: Add OAuth/credential flow for UKDS API

### Future Considerations
- Credential storage: Use keyring or environment variables
- Token refresh: Handle OAuth token expiration
- Access logging: Track which datasets are accessed for compliance
