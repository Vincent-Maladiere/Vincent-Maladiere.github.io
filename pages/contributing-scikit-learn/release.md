# Release

- Changing the version in `VERSION.txt` and `CHANGES.rst`
- Updating tags
    
    ```bash
    git fetch --prune --prune-tags --tags upstream
    git tag x.y.z
    git push upstream x.y.z
    ```
    
- Create wheels and sdist
    
    ```bash
    python setup.py bdist_wheel sdist
    ```
    
- Create a new env, install the wheel just created and run:
    
    ```bash
    pytest --pyargs <package>
    ```
    
    to check that all tests still pass.
    
- Upload to Pypi

```bash
pip install twine -U
twine check dist/*
twine upload dist/* --verbose
```