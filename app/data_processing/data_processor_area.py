from app.data_processing.data_processor import MetricDataProcessor


class DataProcessorArea(MetricDataProcessor):
    """Root-area data processor (shares MetricDataProcessor; area metric column)."""

    value_column = "Area (mm²)"
