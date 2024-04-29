model=$1

if [[ ${model} == "transformer" ]]; then
	mkdir -p ./.data/multi30k
	wget https://github.com/multi30k/dataset/raw/master/data/task1/raw/train.en.gz && mv train.en.gz ./.data/multi30k && gzip -d ./.data/multi30k/train.en.gz
	wget https://github.com/multi30k/dataset/raw/master/data/task1/raw/train.de.gz && mv train.de.gz ./.data/multi30k && gzip -d ./.data/multi30k/train.de.gz
	wget https://github.com/multi30k/dataset/raw/master/data/task1/raw/val.en.gz && mv val.en.gz ./.data/multi30k && gzip -d ./.data/multi30k/val.en.gz
	wget https://github.com/multi30k/dataset/raw/master/data/task1/raw/val.de.gz && mv val.de.gz ./.data/multi30k && gzip -d ./.data/multi30k/val.de.gz
	wget https://github.com/multi30k/dataset/raw/master/data/task1/raw/test_2016_flickr.en.gz && mv test_2016_flickr.en.gz ./.data/multi30k && gzip -d ./.data/multi30k/test_2016_flickr.en.gz
	wget https://github.com/multi30k/dataset/raw/master/data/task1/raw/test_2016_flickr.de.gz && mv test_2016_flickr.de.gz ./.data/multi30k && gzip -d ./.data/multi30k/test_2016_flickr.de.gz

	mv ./.data/multi30k/test_2016_flickr.en ./.data/multi30k/test2016.en
	mv ./.data/multi30k/test_2016_flickr.de ./.data/multi30k/test2016.de
elif [[ ${model} == "bert" ]]; then
	wget http://www.cs.cornell.edu/~cristian/data/cornell_movie_dialogs_corpus.zip
	unzip -qq cornell_movie_dialogs_corpus.zip
	mkdir -p data
	mv cornell\ movie-dialogs\ corpus/movie_conversations.txt ./data
	mv cornell\ movie-dialogs\ corpus/movie_lines.txt ./data
	rm -rf cornell\ movie-dialogs\ corpus
	rm cornell_movie_dialogs_corpus.zip
	rm -rf __MACOSX # optional
else
	echo "other model"
fi