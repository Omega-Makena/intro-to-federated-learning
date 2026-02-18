# Contributing to Intro to Federated Learning

Thank you for your interest in contributing to this project! This guide will help you get started.

## Project Overview

This repository provides a comprehensive introduction to federated learning using the Flower framework, with a practical credit card fraud detection example using PyTorch.

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Git
- Virtual environment (recommended)

### Setup Development Environment

1. **Fork and clone the repository:**
```bash
git clone https://github.com/YOUR_USERNAME/intro-to-federated-learning.git
cd intro-to-federated-learning
```

2. **Create virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Run tests:**
```bash
python test_setup.py
python test_integration.py
```

## Project Structure

```
intro-to-federated-learning/
├── README.md                          # Main documentation
├── QUICKSTART.md                      # Quick start guide
├── CONTRIBUTING.md                    # This file
├── requirements.txt                   # Python dependencies
├── run_federated_learning.sh         # Demo script
├── test_setup.py                      # Setup verification
├── test_integration.py                # Integration tests
│
├── docs/                              # Documentation
│   ├── FLOWER_BASICS.md              # Flower framework guide
│   └── FEDERATED_LEARNING_CONCEPTS.md # FL concepts
│
├── notebooks/                         # Jupyter notebooks
│   ├── 01_flower_basics.ipynb        # Flower tutorial
│   └── 02_federated_learning_demo.ipynb # Complete demo
│
├── src/                               # Source code
│   ├── __init__.py                   # Package init
│   ├── server.py                     # FL server
│   ├── client.py                     # FL client
│   ├── model.py                      # Neural network
│   ├── data_loader.py                # Data utilities
│   └── utils.py                      # Helper functions
│
└── data/                              # Data directory
    └── README.md                     # Dataset info
```

## How to Contribute

### Types of Contributions

1. **Bug Reports**: Found an issue? Report it!
2. **Feature Requests**: Have an idea? Share it!
3. **Documentation**: Improve or add documentation
4. **Code**: Fix bugs or implement features
5. **Examples**: Add new examples or use cases

### Reporting Bugs

When reporting bugs, please include:
- Python version
- Operating system
- Steps to reproduce
- Expected vs. actual behavior
- Error messages (full traceback)
- Relevant code snippets

**Example:**
```
Title: Client fails to connect to server on Windows

Environment:
- Python 3.9.5
- Windows 10
- Flower 1.6.0

Steps to Reproduce:
1. Start server: python src/server.py
2. Start client: python src/client.py --client-id 0
3. Error occurs: [Connection refused]

Error Message:
[Paste full error here]
```

### Suggesting Features

When suggesting features, please include:
- Clear description of the feature
- Use case or motivation
- Expected behavior
- Potential implementation approach (optional)

### Contributing Code

#### 1. Find or Create an Issue

- Check existing issues
- Create new issue if needed
- Discuss approach before major changes

#### 2. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/bug-description
```

#### 3. Make Changes

**Code Style:**
- Follow PEP 8 for Python code
- Use meaningful variable and function names
- Add docstrings to functions and classes
- Keep functions focused and small

**Example:**
```python
def calculate_accuracy(predictions, labels):
    """
    Calculate accuracy of predictions.
    
    Args:
        predictions (torch.Tensor): Model predictions
        labels (torch.Tensor): True labels
    
    Returns:
        float: Accuracy score between 0 and 1
    """
    correct = (predictions == labels).sum().item()
    total = labels.size(0)
    return correct / total
```

#### 4. Test Your Changes

```bash
# Run existing tests
python test_setup.py
python test_integration.py

# Test specific modules
python src/model.py
python src/data_loader.py
python src/utils.py

# Manual testing
python src/server.py  # Terminal 1
python src/client.py --client-id 0  # Terminal 2
```

#### 5. Update Documentation

- Update relevant `.md` files
- Add docstrings to new functions
- Update README.md if needed
- Add examples to notebooks if applicable

#### 6. Commit Changes

Use clear, descriptive commit messages:

```bash
# Good commit messages
git commit -m "Add support for custom learning rates"
git commit -m "Fix client connection timeout issue"
git commit -m "Update documentation for data loading"

# Bad commit messages
git commit -m "fix bug"
git commit -m "update"
git commit -m "changes"
```

#### 7. Submit Pull Request

1. Push to your fork:
```bash
git push origin feature/your-feature-name
```

2. Create PR on GitHub with:
   - Clear title
   - Description of changes
   - Link to related issue
   - Screenshots (if UI changes)
   - Testing performed

**PR Template:**
```markdown
## Description
Brief description of changes

## Related Issue
Fixes #123

## Changes Made
- Change 1
- Change 2
- Change 3

## Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Manual testing completed

## Screenshots (if applicable)
[Add screenshots here]
```

### Contributing Documentation

Documentation improvements are always welcome!

**Areas to improve:**
- Fix typos or unclear explanations
- Add more examples
- Improve existing tutorials
- Translate documentation
- Create video tutorials

**Documentation style:**
- Use clear, simple language
- Provide code examples
- Include visual aids where helpful
- Test all code examples

### Contributing Examples

New examples help others learn!

**Example ideas:**
- Different datasets
- Alternative model architectures
- Custom aggregation strategies
- Privacy-preserving techniques
- Real-world deployment scenarios

**Example structure:**
```python
"""
Example: [Title]

Description: [What this example demonstrates]

Author: [Your Name]
Date: [Date]
"""

# Clear comments throughout
# Working code that can be run as-is
# Expected output documented
```

## Development Guidelines

### Code Quality

- Write clean, readable code
- Follow Python best practices
- Use type hints where appropriate
- Handle errors gracefully
- Log important events

### Testing

- Test new features
- Ensure existing tests pass
- Add tests for bug fixes
- Test edge cases

### Performance

- Consider computational efficiency
- Minimize memory usage
- Profile slow code
- Document performance characteristics

### Security

- Never commit secrets or API keys
- Validate user inputs
- Handle data safely
- Follow security best practices

## Code Review Process

1. Maintainers will review your PR
2. Address feedback and suggestions
3. Make requested changes
4. PR will be merged when approved

**Review timeline:**
- Initial review: 3-5 days
- Follow-up reviews: 1-2 days

## Recognition

Contributors will be:
- Listed in CONTRIBUTORS.md
- Mentioned in release notes
- Credited in commit messages

## Questions?

- Open an issue for questions
- Join discussions on GitHub
- Check existing documentation

## License

By contributing, you agree that your contributions will be licensed under the same license as the project (MIT License).

## Thank You!

Your contributions help make federated learning more accessible to everyone. Thank you for being part of this project!

---

**Happy Contributing! 🎉**
