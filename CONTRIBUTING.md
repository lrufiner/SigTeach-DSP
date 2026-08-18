# Contribuir

```bash
pip install -e ".[dev]"
pytest
ruff check .
```

Las nuevas operaciones deben implementarse primero como funciones puras en
`processing/`, con pruebas, y recién después integrarse en la interfaz.
