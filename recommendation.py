def recommendation(input_subject):
    import pandas as pd
    import numpy as np
    from sentence_transformers import SentenceTransformer, util

    model = SentenceTransformer('stsb-roberta-large')

    df = pd.read_csv('Unisen School Datas - Curriculum CBSE.csv', header=0).dropna()
    subjects_list = df['Class 12'].tolist()
    vectors = model.encode(subjects_list)
    subjects_vectors = dict(zip(df['Class 12'], vectors))

    if len(input_subject) == 1:
        input_vector = model.encode(input_subject[0])
        similarities = {}
        for subject, vector in subjects_vectors.items():
            if subject not in input_subject:  
                sim = util.cos_sim([input_vector], [vector])[0][0]
                similarities[subject] = sim
        similar_subjects = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[0:6]
        for subject in similar_subjects:
            print(f"{subject[0]} : {subject[1]:.4f}")
    else:
        input_vectors = model.encode(input_subject)
        mean_vector = np.mean(input_vectors, axis=0)
        similarities = {}
        for subject, vector in subjects_vectors.items():
            if subject not in input_subject:  
                sim = util.cos_sim([mean_vector], [vector])[0][0]
                similarities[subject] = sim
        similar_subjects = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[0:6]
        for subject in similar_subjects:
            print(f"{subject[0]} : {subject[1]:.4f}")

input_subject=['Mathematics']
test_cases=['Physics','Chemistry','Biology','Economics']

for subject in test_cases:
    new_input_subject = input_subject + [subject]
    print('Test Case',subject)
    recommendation(new_input_subject)
