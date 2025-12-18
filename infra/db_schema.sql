-- Database schema for Property Valuation AVM
-- PostgreSQL schema

-- Properties table
CREATE TABLE IF NOT EXISTS properties (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Numeric features
    bedrooms INTEGER NOT NULL CHECK (bedrooms > 0),
    bathrooms NUMERIC(3, 1) NOT NULL CHECK (bathrooms > 0),
    sqft_living INTEGER NOT NULL CHECK (sqft_living > 0),
    sqft_lot INTEGER NOT NULL CHECK (sqft_lot >= 0),
    floors NUMERIC(2, 1) NOT NULL CHECK (floors > 0),
    year_built INTEGER NOT NULL CHECK (year_built >= 1800),
    year_renovated INTEGER DEFAULT 0 CHECK (year_renovated >= 0),

    -- Location features
    latitude NUMERIC(10, 8) NOT NULL,
    longitude NUMERIC(11, 8) NOT NULL,

    -- Categorical features
    property_type VARCHAR(50) NOT NULL,
    neighborhood VARCHAR(100) NOT NULL,
    condition VARCHAR(20) NOT NULL,
    view_quality VARCHAR(20) NOT NULL,

    -- Text and media
    description TEXT,
    image_path VARCHAR(500),

    -- Prices
    actual_price NUMERIC(12, 2),
    predicted_price NUMERIC(12, 2),

    -- Metadata
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    -- Constraints
    CONSTRAINT valid_property_type CHECK (
        property_type IN ('Single Family', 'Townhouse', 'Condo', 'Multi-Family')
    ),
    CONSTRAINT valid_condition CHECK (
        condition IN ('Poor', 'Fair', 'Average', 'Good', 'Excellent')
    ),
    CONSTRAINT valid_view_quality CHECK (
        view_quality IN ('None', 'Fair', 'Good', 'Excellent')
    )
);

-- Predictions table (history of predictions)
CREATE TABLE IF NOT EXISTS predictions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    property_id UUID REFERENCES properties(id) ON DELETE CASCADE,

    -- Prediction details
    predicted_price NUMERIC(12, 2) NOT NULL,
    confidence_score NUMERIC(4, 3) CHECK (confidence_score >= 0 AND confidence_score <= 1),

    -- Model breakdown
    tabular_prediction NUMERIC(12, 2),
    image_prediction NUMERIC(12, 2),
    text_prediction NUMERIC(12, 2),

    -- Model metadata
    model_version VARCHAR(50),
    ensemble_used BOOLEAN DEFAULT TRUE,

    -- Timestamp
    predicted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    -- API metadata
    api_version VARCHAR(20),
    request_id VARCHAR(100)
);

-- Model performance metrics table
CREATE TABLE IF NOT EXISTS model_metrics (
    id SERIAL PRIMARY KEY,
    model_name VARCHAR(50) NOT NULL,
    model_version VARCHAR(50) NOT NULL,

    -- Metrics
    mae NUMERIC(12, 2),
    mape NUMERIC(5, 2),
    rmse NUMERIC(12, 2),
    r2_score NUMERIC(5, 4),

    -- Training info
    training_samples INTEGER,
    validation_samples INTEGER,
    training_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    -- Additional metadata
    hyperparameters JSONB,
    notes TEXT
);

-- User feedback table (for continuous improvement)
CREATE TABLE IF NOT EXISTS prediction_feedback (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    prediction_id UUID REFERENCES predictions(id) ON DELETE CASCADE,

    -- Feedback
    actual_sale_price NUMERIC(12, 2),
    user_rating INTEGER CHECK (user_rating >= 1 AND user_rating <= 5),
    feedback_text TEXT,

    -- Metadata
    submitted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    submitted_by VARCHAR(100)
);

-- Create indexes for performance
CREATE INDEX idx_properties_neighborhood ON properties(neighborhood);
CREATE INDEX idx_properties_property_type ON properties(property_type);
CREATE INDEX idx_properties_created_at ON properties(created_at);
CREATE INDEX idx_properties_actual_price ON properties(actual_price);
CREATE INDEX idx_predictions_property_id ON predictions(property_id);
CREATE INDEX idx_predictions_predicted_at ON predictions(predicted_at);
CREATE INDEX idx_model_metrics_model_name ON model_metrics(model_name);

-- Create updated_at trigger
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_properties_updated_at
    BEFORE UPDATE ON properties
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Create views for analytics
CREATE OR REPLACE VIEW property_valuation_accuracy AS
SELECT
    p.id,
    p.property_type,
    p.neighborhood,
    p.actual_price,
    pred.predicted_price,
    ABS(p.actual_price - pred.predicted_price) as price_difference,
    ABS(p.actual_price - pred.predicted_price) / p.actual_price * 100 as percentage_error,
    pred.confidence_score,
    pred.predicted_at
FROM properties p
JOIN predictions pred ON p.id = pred.property_id
WHERE p.actual_price IS NOT NULL;

CREATE OR REPLACE VIEW neighborhood_stats AS
SELECT
    neighborhood,
    COUNT(*) as property_count,
    AVG(actual_price) as avg_price,
    MIN(actual_price) as min_price,
    MAX(actual_price) as max_price,
    STDDEV(actual_price) as price_stddev,
    AVG(bedrooms) as avg_bedrooms,
    AVG(sqft_living) as avg_sqft
FROM properties
WHERE actual_price IS NOT NULL
GROUP BY neighborhood;

-- Insert sample model metrics
INSERT INTO model_metrics (model_name, model_version, mae, mape, r2_score, training_samples, notes)
VALUES
    ('tabular', '1.0.0', 25000, 8.5, 0.92, 10000, 'Initial gradient boosting model'),
    ('image', '1.0.0', 35000, 12.0, 0.85, 10000, 'ResNet50-based image model'),
    ('text', '1.0.0', 40000, 14.0, 0.82, 10000, 'BERT-based text model'),
    ('stacker', '1.0.0', 20000, 6.5, 0.95, 10000, 'Ensemble stacking model')
ON CONFLICT DO NOTHING;

-- Comments for documentation
COMMENT ON TABLE properties IS 'Main table storing property information and features';
COMMENT ON TABLE predictions IS 'Historical record of all predictions made by the AVM';
COMMENT ON TABLE model_metrics IS 'Performance metrics for different model versions';
COMMENT ON TABLE prediction_feedback IS 'User feedback on prediction accuracy for model improvement';
COMMENT ON VIEW property_valuation_accuracy IS 'View showing prediction accuracy metrics';
COMMENT ON VIEW neighborhood_stats IS 'Aggregated statistics by neighborhood';
