from modular_functions import generate_metric_embedding_from_file, load_metric_embedding, \
    split_arabic_text_into_sentences, clean_arabic, get_embedding, l2_normalize
from tqdm import tqdm






"""
This function takes a list of metric names and generates embeddings for each metric
by reading the metric files and loading them into memory.
"""
def add_metric_list_embeddings(list_metrics):
    for metric in tqdm(list_metrics, desc="Generating metric embeddings"):
        generate_metric_embedding_from_file(metric)
        load_metric_embedding(metric)

"""
This function tests a sentence-splitting function by applying it to several
sample Arabic paragraphs and printing the resulting sentences.
"""
def sentence_splitter_tester(function):
    sentence_1 = "اعترض زوج المريضةأنه بتاريخ 9-1-2025 حضروا إلى مستشفانا (وقوع المريضة على يدها)للقيام بإجراءات ما قبل الدخول وإدخال المريضة (بتاريخ 10-1-2025) وإجراء العملية(سياخ وبراغي)لها لدى د.محمد باقر عزالدين واعتراضهم حول الإنتظار الطويل في قسم وحدة ما قبل الدخول (حوالي الساعة من الوقت) إضافة إلى عدم وجود آلية محددة بموضوع الأدوار"
    sentence_2 = "اعترض ابن المريض صالح يوسف صالح انه بتاريخ 4 3 2025 ادخل الي المستشفي حوالي س 8 00 ليلا الي الثالث شرقي غرفه 321 عمليه لدي د اسعد منصور وفي اليوم التالي اخبروه انه لا يوجد حجز له فتفاجا واتصل بالطبيب الذي اخبره انه لم يعلم بدخوله خاصه انه تم اعلامه ان العمليه هي س 10 30 حينها فكيف ذلك"
    sentence_3 = "اعترضت ابنه المريضه نورالهدي عبدالعفو القاطرجي انه بتاريخ 22 2 2025 حضرت المريضه الي الطواري التهابات الروايا ومن ثم نقلها الي الرابع غربي غرفه 409 وانه 1 انه بتاريخ الاثنين 24 2 2025 حوالي س 10 00 ليلا حضرت الممرضه وارادت وضع دواء بال للمريضه لكن بسبب انشغالهم بالتغيير للمريضه قالت الممرضه ان تناديها بعد الانتهاء لتضع الدواء فقامت المرافقه بحمل الدواء للتاكد من الاسم المرافقه ممرضه وارادت هي ان تضع الدواء فقامت بقراءه اسم مريضه ثانيه علي الدواء وليس مريضتها 2 ان الاثنين بتاريخ 24 2 2025 لم يتم اعطاء دواء المهدي للمريضه فذهبت المرافقه وسالت الممرضه ، فاجابتها الممرضه انها انشغلت ونسيت كما ذكرت الامر الذي لم تتقبله المرافقه"
    sentence_4 = " اعترض محمد هشام مراد حول تواصل موظف الدخول معهم بتاريخ 9 5 2025 عمليه كسر في اليد د اسعد منصور وابلاغهم بعدم الحضور حتي تامين سرير"

    sentences = [sentence_1, sentence_2, sentence_3, sentence_4]
    for idx, sentence in enumerate(sentences, start=1):
        print(f"--- Processing Paragraph {idx} ---")
        paragraph = function(sentence)
        for i, sen in enumerate(paragraph, start=1): print(f"Sentence {i}: {sen}")
        print("End of this paragraph")
        print("--------------------------------------------")


""""
This function here, splits the patient_feedback in max of 6 sentences and saves 
their embeddings there. 
"""
def save_sentence_embeddings_to_db():
    import os
    import sqlite3
    from tqdm import tqdm   # ← progress bar

    DB_PATH = os.path.join("../..", "patient_feedback_ml.db")
    TABLE = "patient_feedback_encoded"

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # 1️⃣ Ensure the 6 embedding columns exist
    for i in range(1, 7):
        try:
            cursor.execute(f"""
                ALTER TABLE {TABLE}
                ADD COLUMN sentence_{i}_embedding BLOB;
            """)
        except sqlite3.OperationalError:
            pass

    conn.commit()

    # 2️⃣ Load all rows
    cursor.execute(f"""
        SELECT rowid, complaint_text 
        FROM {TABLE}
    """)
    rows = cursor.fetchall()
    print(f"Processing {len(rows)} records...")

    # 3️⃣ Progress bar here
    for rowid, text in tqdm(rows, desc="Encoding sentences"):

        if text is None or str(text).strip() == "":
            continue

        # Split → max 6 sentences
        sentences = split_arabic_text_into_sentences(text)[:6]

        # Clean them
        sentences = [clean_arabic(s) for s in sentences]

        # Generate embeddings for each sentence
        embeddings = []
        for s in sentences:
            emb = get_embedding(s)   # ⬅ returns vector
            embeddings.append(emb)

        # If fewer than 6 sentences → padding
        while len(embeddings) < 6:
            embeddings.append(None)

        # Update only embedding columns
        cursor.execute(
            f"""
            UPDATE {TABLE}
            SET
                sentence_1_embedding=?,
                sentence_2_embedding=?,
                sentence_3_embedding=?,
                sentence_4_embedding=?,
                sentence_5_embedding=?,
                sentence_6_embedding=?
            WHERE rowid=?
            """,
            (
                embeddings[0],
                embeddings[1],
                embeddings[2],
                embeddings[3],
                embeddings[4],
                embeddings[5],
                rowid,
            )
        )

    conn.commit()
    conn.close()
    print("Done.")


"""
This function recreates train and test tables from a source table by shuffling
the rows and splitting according to a specified test ratio.
"""
def recreate_train_test_tables(conn, source_table, train_table, test_table, test_ratio=0.2):
    cur = conn.cursor()

    # Drop old tables
    cur.execute(f"DROP TABLE IF EXISTS {train_table}")
    cur.execute(f"DROP TABLE IF EXISTS {test_table}")

    # Read all data
    cur.execute(f"SELECT * FROM {source_table}")
    rows = cur.fetchall()

    import random
    random.shuffle(rows)

    cutoff = int(len(rows) * (1 - test_ratio))
    train_rows = rows[:cutoff]
    test_rows = rows[cutoff:]

    # Create tables with the same schema
    cur.execute(f"CREATE TABLE {train_table} AS SELECT * FROM {source_table} WHERE 0")
    cur.execute(f"CREATE TABLE {test_table} AS SELECT * FROM {source_table} WHERE 0")

    # Insert split data
    placeholders = ",".join(["?"] * len(train_rows[0]))

    cur.executemany(
        f"INSERT INTO {train_table} VALUES ({placeholders})", train_rows
    )

    cur.executemany(
        f"INSERT INTO {test_table} VALUES ({placeholders})", test_rows
    )

    conn.commit()


"""
This function normalizes all embeddings stored in the database (both sentence-level
and text-level) so that each vector has unit length for proper similarity computation.
"""
def normalize_database_embeddings():
    import os
    import sqlite3
    import numpy as np
    import ast
    from tqdm import tqdm  # progress bar

    DB_PATH = os.path.join("../..", "patient_feedback_ml.db")
    TABLE = "patient_feedback_encoded"

    embedding_columns = [
        "sentence_1_embedding",
        "sentence_2_embedding",
        "sentence_3_embedding",
        "sentence_4_embedding",
        "sentence_5_embedding",
        "sentence_6_embedding",
        "embedding_text1",
        "embedding_text2",
        "embedding_text3",
        "embedding_text123",
        "embedding_text23"
    ]

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute(f"""
        SELECT rowid, {",".join(embedding_columns)}
        FROM {TABLE}
    """)
    rows = cur.fetchall()

    print(f"[INFO] Normalizing {len(rows)} records from the DB...")

    for row in tqdm(rows):
        rowid = row[0]
        col_values = row[1:]

        new_values = []
        for emb in col_values:

            if emb is None:
                new_values.append(None)
                continue

            # ------------------------------------------
            # CASE A: embedding stored as raw bytes (correct)
            # ------------------------------------------
            if isinstance(emb, (bytes, bytearray)):
                vec = np.frombuffer(emb, dtype=np.float32)

            # ------------------------------------------
            # CASE B: embedding stored as string (old version)
            # ------------------------------------------
            elif isinstance(emb, str):
                try:
                    # Try to parse as Python list string
                    vec_list = ast.literal_eval(emb)
                    vec = np.array(vec_list, dtype=np.float32)
                except:
                    # Last fallback: comma split
                    vec = np.array([float(x) for x in emb.split(',')], dtype=np.float32)

            else:
                new_values.append(None)
                continue

            # Normalize
            norm = np.linalg.norm(vec)
            if norm == 0:
                new_vec = vec
            else:
                new_vec = vec / norm

            new_values.append(new_vec.astype(np.float32).tobytes())

        set_clause = ", ".join([f"{col}=?" for col in embedding_columns])

        cur.execute(
            f"UPDATE {TABLE} SET {set_clause} WHERE rowid=?",
            (*new_values, rowid)
        )

    conn.commit()
    conn.close()
    print("[DONE] Database embeddings normalized.")



"""
This function generates embeddings for the main text columns in the database:
- embedding_text1  : embedding of 'complaint_text'
- embedding_text2  : embedding of 'immediate_action'
- embedding_text3  : embedding of 'taken_action'
- embedding_text123: embedding of the combined text of 'complaint_text', 'immediate_action', and 'taken_action'
- embedding_text23 : embedding of the combined text of 'immediate_action' and 'taken_action'

It cleans each text using 'clean_arabic', computes embeddings using 'get_embedding',
and updates the 'patient_feedback_encoded' table with the resulting vectors.
"""
def save_text_column_embeddings_to_db():
    """
    Generates embeddings for three text columns and their combinations:
    - embedding_text1  : complaint_text
    - embedding_text2  : immediate_action
    - embedding_text3  : taken_action
    - embedding_text123: complaint_text + immediate_action + taken_action
    - embedding_text23 : immediate_action + taken_action
    Stores them in the 'patient_feedback_encoded' table.
    """
    import os
    import sqlite3
    from tqdm import tqdm

    DB_PATH = os.path.join("../..", "patient_feedback_ml.db")
    TABLE = "patient_feedback_encoded"

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Ensure embedding columns exist
    text_columns = [
        "embedding_text1",
        "embedding_text2",
        "embedding_text3",
        "embedding_text123",
        "embedding_text23"
    ]
    for col in text_columns:
        try:
            cursor.execute(f"ALTER TABLE {TABLE} ADD COLUMN {col} BLOB;")
        except sqlite3.OperationalError:
            pass  # column already exists

    conn.commit()

    # Load all rows
    cursor.execute(f"SELECT rowid, complaint_text, immediate_action, taken_action FROM {TABLE}")
    rows = cursor.fetchall()
    print(f"Processing {len(rows)} records for text-column embeddings...")

    for rowid, complaint_text, immediate_action, taken_action in tqdm(rows, desc="Generating embeddings"):
        # Clean texts
        text1 = clean_arabic(complaint_text) if complaint_text else ""
        text2 = clean_arabic(immediate_action) if immediate_action else ""
        text3 = clean_arabic(taken_action) if taken_action else ""

        # Individual embeddings
        emb1 = get_embedding(text1) if text1 else None
        emb2 = get_embedding(text2) if text2 else None
        emb3 = get_embedding(text3) if text3 else None

        # Combined embeddings
        emb123 = get_embedding(" ".join([text1, text2, text3])) if text1 or text2 or text3 else None
        emb23 = get_embedding(" ".join([text2, text3])) if text2 or text3 else None

        # Update DB
        cursor.execute(
            f"""
            UPDATE {TABLE}
            SET
                embedding_text1=?,
                embedding_text2=?,
                embedding_text3=?,
                embedding_text123=?,
                embedding_text23=?
            WHERE rowid=?
            """,
            (emb1, emb2, emb3, emb123, emb23, rowid)
        )

    conn.commit()
    conn.close()
    print("[DONE] Text-column embeddings saved to database.")





def Embedd_Normalize(metric_name, folder="vocab_dataset"):
    import os
    import json
    from tqdm import tqdm
    import numpy as np

    input_path = os.path.join(folder, f"{metric_name}.json")
    output_path = os.path.join(folder, f"{metric_name}_normalized.json")

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    sentences = []
    labels = []

    # -----------------------------------------------------
    # CASE 1: Normal expected format {positive:[], negative:[]}
    # -----------------------------------------------------
    if isinstance(data, dict):
        for text in data.get("positive", []):
            sentences.append(text)
            labels.append(1)

        for text in data.get("negative", []):
            sentences.append(text)
            labels.append(0)

    # -----------------------------------------------------
    # CASE 2: Data is a flat list → treat all as positive
    # -----------------------------------------------------
    elif isinstance(data, list):
        for text in data:
            sentences.append(text)
            labels.append(1)   # or 0 if needed

    else:
        raise ValueError("Unsupported JSON structure")

    print(f"[INFO] Loaded {len(sentences)} sentences")

    normalized_vectors = []

    for text in tqdm(sentences, desc=f"Embedding {metric_name}", unit="item"):
        emb = get_embedding(text)
        emb = np.frombuffer(emb, dtype=np.float32) if isinstance(emb, bytes) else emb
        emb_norm = l2_normalize(emb)
        normalized_vectors.append(emb_norm.tolist())

    result = {
        "vectors": normalized_vectors,
        "labels": labels,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=4)

    print(f"[DONE] Saved normalized embeddings to: {output_path}")





if __name__ == "__main__":
    print("Sameer")
