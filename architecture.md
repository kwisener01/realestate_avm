property-valuation/
  app/
    main.py              # FastAPI entry
    api/
      routes_predict.py
      routes_properties.py
    models/              # Pydantic API models
  ml/
    image_model.py
    text_model.py
    tabular_model.py
    stacker.py
    train_image.py
    train_text.py
    train_tabular.py
    train_stack.py
  data/
    raw/
    processed/
  scripts/
    prepare_dataset.py
    migrate_db.py
  infra/
    docker-compose.yml
    db_schema.sql
