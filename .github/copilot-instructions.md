# Copilot Instructions — Entity Linker 🔍🔗

Instrucciones y estándares para el desarrollo asistido por IA (Claude Code, GitHub Copilot) en el proyecto **Entity Linker**.

---

## 1. Enunciado y Objetivos de la Prueba Técnica (Abierto)

El proyecto consiste en diseñar e implementar un servicio web funcional de **Reconocimiento y Enlace de Entidades (Entity Linking estilo Argos)** a Wikidata.

### Reto Principal
- **Entrada:** Texto en lenguaje natural.
- **Procesamiento:**
  1. Identificar las entidades nombradas presentes en el texto (personas, organizaciones, lugares, obras u otras categorías relevantes).
  2. Enlazar cada entidad detectada con su elemento correspondiente en Wikidata (QID), resolviendo los casos de ambigüedad en los que varias entidades compartan nombre o forma superficial.
- **Salida:** Servicio web funcional, ejecutable y verificable en vivo (por ejemplo, mediante una interfaz autogenerada Swagger UI / OpenAPI u otra solución REST sencilla).

### Enfoque Abierto y Evolutivo
Para mantener la máxima flexibilidad durante las fases de prototipado y ajuste, **las decisiones técnicas concretas sobre el pipeline no están congeladas de antemano**. El diseño exacto de los componentes (modelos de NER, estrategias de consulta a la API de Wikidata, algoritmos de desambiguación, heurísticas de scoring o reglas para asignar `NIL`) se irá especificando y refinando progresivamente.

### Apartados a Desarrollar en la Memoria / Entrega
El código y la arquitectura deben permitir defender y documentar los siguientes puntos clave:
1. **Alcance y casos de uso:** Definición del problema, tipos de entidad, idioma de los textos, volumen esperado y nivel de ambigüedad.
2. **Arquitectura de la solución:** Propuesta del pipeline general, técnicas empleadas en cada fase y justificación frente a alternativas razonables.
3. **Resolución de la ambigüedad:** Señales de contexto y heurísticas utilizadas para seleccionar el elemento correcto en Wikidata entre múltiples candidatos.
4. **Evaluación y casos límite:** Métricas de calidad y tratamiento de entidades inexistentes en Wikidata (`NIL`), poco conocidas o ambiguas.
5. **Optimización si se usaran LLMs propios:** Propuestas de mejora en velocidad, latencia y cómputo frente al uso de APIs de terceros.
6. **Producción y mantenimiento:** Estrategias de escalado, costes, latencia y actualización continua ante cambios en Wikidata.

---

## 2. Environment & Dependency Management

- **Versión de Python:** Python 3.12+ fijado explícitamente en `.python-version`.
- **Gestor de Paquetes:** Uso exclusivo de **`uv`**.
  - Sincronizar entorno: `uv sync`
  - Ejecutar comandos o herramientas: `uv run uvicorn ...`, `uv run pytest`
  - Añadir dependencias: `uv add <paquete>` o `uv add --dev <paquete>`
- **Control de Versiones:** Mantener en Git los archivos `.python-version`, `pyproject.toml` y `uv.lock` para garantizar ejecuciones 100% reproducibles.

---

## 3. Estándares de Código Python

### Estilo y Formato
- **Cumplimiento estricto de PEP 8:** Indentación de 4 espacios, `snake_case` para funciones/variables, `PascalCase` para clases y `UPPER_SNAKE_CASE` para constantes.
- **Límite de longitud de línea:** Máximo 88 caracteres por línea.

### Tipado Moderno (Python 3.10+)
- Usar tipos y sintaxis nativas de colecciones y uniones:
  - `list[str]` en lugar de `typing.List[str]`
  - `dict[str, Any]` en lugar de `typing.Dict[str, Any]`
  - `str | None` en lugar de `typing.Optional[str]`

### Política Pragmática de Docstrings
- **Código Autoexplicativo Primero:** Priorizar nombres de funciones expresivos, funciones pequeñas y anotaciones de tipo claras.
- **Omitir Docstrings:** En funciones auxiliares privadas, triviales o internas donde el nombre y las anotaciones de tipo explican completamente el comportamiento.
- **Docstrings Obligatorios (Google Style):** En módulos públicos, endpoints de la API, clases de dominio principal, capas de servicio e interfaces complejas donde sea necesario explicar la lógica de negocio, excepciones lanzadas (`Raises:`) o comportamientos no evidentes.

### Modelos de Datos con Pydantic v2
- Definir esquemas heredando de `pydantic.BaseModel`.
- Incluir descripciones claras mediante `Field(..., description="...")` para enriquecer la documentación OpenAPI.
- Utilizar `model.model_dump()` en lugar del método obsoleto `.dict()`.

### Gestión de Red y Excepciones
- **API Politeness:** Incluir siempre una cabecera `User-Agent` personalizada y descriptiva en todas las peticiones externas dirigidas a Wikidata.
- **Resiliencia:** Capturar excepciones específicas (`httpx.HTTPError`, `ValueError`). Las rutas de FastAPI deben elevar `HTTPException` con códigos de estado adecuados.

---

## 4. Guía de Pruebas Unitarias

- **Framework:** `pytest` + `pytest-asyncio` con configuración `asyncio_mode = "auto"`.
- **Ubicación:** Directorio `tests/` en la raíz del proyecto.
- **Aislamiento de Red:** Los tests unitarios e integración deben utilizar *mocks* de red (`unittest.mock.patch` o `respx`) para evitar llamadas reales a Wikidata durante la ejecución de los tests. Esto asegura que la suite de pruebas sea determinista, ultrarrápida y ejecutable sin conexión a internet.

---

## 5. Convenciones de Git y Flujo de Trabajo

- **Conventional Commits:** Formatear los mensajes de commit indicando el tipo de cambio:
  - `feat:` nuevas funcionalidades
  - `fix:` corrección de errores
  - `test:` pruebas unitarias o de integración
  - `docs:` cambios en documentación
  - `refactor:` mejoras internas de código sin cambiar comportamiento
