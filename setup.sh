mkdir logs
mkdir outputs
mkdir datasets
mkdir cache

echo Downloading datasets
wget http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_CDs_and_Vinyl_5.json.gz
wget http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Clothing_Shoes_and_Jewelry_5.json.gz
wget http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Toys_and_Games_5.json.gz
wget http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Video_Games_5.json.gz
gzip -d reviews_CDs_and_Vinyl_5.json.gz
gzip -d reviews_Clothing_Shoes_and_Jewelry_5.json.gz
gzip -d reviews_Toys_and_Games_5.json.gz
gzip -d reviews_Video_Games_5.json.gz
mv reviews_CDs_and_Vinyl_5.json ./datasets/
mv reviews_Clothing_Shoes_and_Jewelry_5.json ./datasets/
mv reviews_Toys_and_Games_5.json ./datasets/
mv reviews_Video_Games_5.json ./datasets/

echo Setup environmen
pip install nltk
pip install torch --extra-index-url https://download.pytorch.org/whl/cu113
pip install scikit-learn==1.0
pip install transformers
python -c "import nltk; nltk.download('stopwords')"
