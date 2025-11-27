
from models_directory.Classification_Models.Hierarchical_Classification_Model.domain.train_domain_model import train_domain_models
from models_directory.Classification_Models.Hierarchical_Classification_Model.category.domain_1.train_category_domain1 import train_category_domain1
from models_directory.Classification_Models.Hierarchical_Classification_Model.category.domain_2.train_category_domain2 import train_category_domain2
from models_directory.Classification_Models.Hierarchical_Classification_Model.category.domain_3.train_category_domain3 import train_category_domain3
from models_directory.Classification_Models.Hierarchical_Classification_Model.sub_category.category_1.train_subcategory_category1 import train_subcategory_cat1
from models_directory.Classification_Models.Hierarchical_Classification_Model.sub_category.category_2.train_subcategory_category2 import train_subcategory_cat2
from models_directory.Classification_Models.Hierarchical_Classification_Model.sub_category.category_3.train_subcategory_category3 import train_subcategory_cat3
from models_directory.Classification_Models.Hierarchical_Classification_Model.sub_category.category_4.train_subcategory_category4 import train_subcategory_cat4
from models_directory.Classification_Models.Hierarchical_Classification_Model.sub_category.category_5.train_subcategory_category5 import train_subcategory_cat5
from models_directory.Classification_Models.Hierarchical_Classification_Model.sub_category.category_6.train_subcategory_category6 import train_subcategory_cat6
from models_directory.Classification_Models.Hierarchical_Classification_Model.sub_category.category_7.train_subcategory_category7 import train_subcategory_cat7
from models_directory.Classification_Models.Harm_level.train_harm_binary import train_harm_binary
from models_directory.Classification_Models.Harm_level.train_harm_ordinal_high import train_harm_ordinal_high
from models_directory.Classification_Models.Harm_level.train_harm_ordinal_low import train_harm_ordinal_low
from models_directory.Classification_Models.Severity_level.train_severity_model import train_severity_model



import json

from models_directory.Classification_Models.Stage.train_stage import train_stage

#What does this script needs to do

#Fetch from the database (For now not existing) the new data



if __name__ == "__main__":
    models_1, metrics_1 = train_category_domain1()
    print(json.dumps(metrics_1, indent=4))
    models_2, metrics_2 = train_category_domain2()
    print(json.dumps(metrics_2, indent=4))
    models_3, metrics_3 = train_category_domain3()
    print(json.dumps(metrics_3, indent=4))
    models_4, metrics_4 = train_domain_models()
    print(json.dumps(metrics_4, indent=4))
    models_5 , metrics_5 = train_subcategory_cat1()
    print(json.dumps(metrics_5, indent=4))
    models_6, metrics_6 = train_subcategory_cat2()
    print(json.dumps(metrics_6, indent=4))
    models_7, metrics_7 = train_subcategory_cat3()
    print(json.dumps(metrics_7, indent=4))
    model_8, metrics_8 = train_subcategory_cat4()
    print(json.dumps(metrics_8, indent=4))
    model_9, metrics_9 = train_subcategory_cat5()
    print(json.dumps(metrics_9, indent=4))
    model_10 ,metrics_10 = train_subcategory_cat6()
    print(json.dumps(metrics_10, indent=4))
    model_11, metrics_11 = train_subcategory_cat7()
    print(json.dumps(metrics_11, indent=4))
    model_12 , metrics_12 = train_harm_binary()
    print(json.dumps(metrics_12, indent=4))
    model_13, metrics_13 = train_harm_ordinal_high()
    print(json.dumps(metrics_13, indent=4))
    model_14, metrics_14 = train_harm_ordinal_low()
    print(json.dumps(metrics_14, indent=4))
    model_15, metrics_15 = train_severity_model()
    print(json.dumps(metrics_15, indent=4))

    # train_stage()




