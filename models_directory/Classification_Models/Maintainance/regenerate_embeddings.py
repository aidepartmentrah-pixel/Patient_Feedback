from models_directory.Classification_Models.Stage.execution_operations import save_text_column_embeddings_to_db, \
    save_sentence_embeddings_to_db, normalize_database_embeddings, recreate_train_test_tables



import sqlite3

conn = sqlite3.connect("../../patient_feedback_ml.db")


save_text_column_embeddings_to_db()
save_sentence_embeddings_to_db()
normalize_database_embeddings()
recreate_train_test_tables(
    conn=conn,
    source_table="patient_feedback_encoded",
    train_table="train_feedback",
    test_table="test_feedback",
    test_ratio=0.2
)

conn.close()
