# Security Policy

## Supported Versions

This project uses the following security-patched versions:

| Dependency | Version | Security Status |
| ---------- | ------- | --------------- |
| torch      | >=2.6.0 |  Patched      |
| torchvision| >=0.21.0|  Compatible   |
| flwr       | 1.6.0   |  Current      |

## Known Vulnerabilities (Addressed)

### PyTorch < 2.6.0

The following vulnerabilities existed in PyTorch versions prior to 2.6.0:

1. **Heap buffer overflow** (CVE pending)
   - Affected: torch < 2.2.0
   - Fixed in: 2.2.0
   - Status:  Addressed (using >=2.6.0)

2. **Use-after-free vulnerability** (CVE pending)
   - Affected: torch < 2.2.0
   - Fixed in: 2.2.0
   - Status:  Addressed (using >=2.6.0)

3. **Remote code execution via torch.load** (CVE pending)
   - Affected: torch < 2.6.0
   - Fixed in: 2.6.0
   - Status:  Addressed (using >=2.6.0)
   - Mitigation: Always use `weights_only=True` when loading models

4. **Deserialization vulnerability** (WITHDRAWN ADVISORY)
   - Affected: torch <= 2.3.1
   - Status: Advisory withdrawn, patched in 2.6.0
   - Status:  Addressed (using >=2.6.0)

## Security Best Practices

### When Using This Repository

1. **Always use the latest versions**
   ```bash
   pip install -r requirements.txt --upgrade
   ```

2. **Model Loading**
   - When loading PyTorch models, always use `weights_only=True`:
   ```python
   torch.load("model.pth", weights_only=True)
   ```

3. **Data Privacy**
   - This federated learning implementation keeps data local
   - Only model updates are transmitted
   - Never share raw data with the server

4. **Network Security**
   - In production, use TLS/SSL for client-server communication
   - Configure Flower with secure connections
   - Use authentication mechanisms

### Updating Dependencies

To update to the latest secure versions:

```bash
# Update all dependencies
pip install -r requirements.txt --upgrade

# Or update specific packages
pip install --upgrade torch>=2.6.0
pip install --upgrade torchvision>=0.21.0
```

## Reporting Security Issues

If you discover a security vulnerability in this project:

1. **Do NOT** open a public issue
2. Email the maintainers directly (see repository owner)
3. Include:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if any)

## Security Checklist

Before deploying to production:

- [ ] All dependencies are up to date
- [ ] PyTorch >= 2.6.0 installed
- [ ] Model loading uses `weights_only=True`
- [ ] TLS/SSL configured for network communication
- [ ] Authentication implemented
- [ ] Data validation in place
- [ ] Error handling properly implemented
- [ ] Logs don't contain sensitive data
- [ ] Input sanitization implemented

## Additional Resources

- [PyTorch Security](https://pytorch.org/docs/stable/notes/security.html)
- [Flower Security Best Practices](https://flower.dev/docs/)
- [OWASP Security Guidelines](https://owasp.org/)

## Version History

| Date | Version | Security Changes |
|------|---------|------------------|
| 2026-02-17 | 1.0.0 | Initial release with PyTorch >=2.6.0 |

## Contact

For security concerns, please contact the repository maintainers.

---

**Last Updated**: 2026-02-17  
**Status**:  All known vulnerabilities addressed
