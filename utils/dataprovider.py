import json
import os
import numpy as np
# from gensim.models.word2vec import Word2Vec
import cPickle
import config


class DataProvider():
    def __init__(
        self,
        batch_size,
        video_len,
        video_region_num,
        video_feature_dim,
        before_len,
        after_len,
        video_dir,
        dataset_dir,
        word2vec_dir,
        wordemb_dim
    ):
        self.batch_size = batch_size
        self.video_len = video_len
        self.video_region_num = video_region_num
        self.video_feature_dim = video_feature_dim
        self.before_len = before_len
        self.after_len = after_len
        self.video_dir = video_dir
        self.dataset_dir = dataset_dir
        self.word2vec_dir = word2vec_dir
        self.word2vecModel = cPickle.load(
            open(os.path.join(self.word2vec_dir, 'simpleModel.pkl'), 'rb'),
        )
        self.woremb_dim = wordemb_dim

    def load_dataset(self, datatype='train'):
        dataset = json.load(
            open(os.path.join(self.dataset_dir, datatype+'_pro.json'), 'r')
        )
        return dataset

    def encodeSentence(self, sentence, texttype='before'):
        # print sentence
        sentence_f = []
        for word in sentence:
            try:
                wordvec = self.word2vecModel[word]
            except KeyError:
                wordvec = self.word2vecModel['UNK']
            sentence_f.append(wordvec)

        if texttype == 'before':
            text_len_var = self.before_len
        else:
            text_len_var = self.after_len
        length = len(sentence_f)
        if length == 0:
            text_data = np.zeros((text_len_var, self.woremb_dim))
            text_mask = np.zeros((text_len_var, ))
        elif length < text_len_var:
            sentence_f = np.array(sentence_f, dtype='float32')
            text_data = np.concatenate(
                [sentence_f, np.zeros((text_len_var-length, self.woremb_dim))],
                axis=0
            )
            text_mask = np.concatenate(
                [np.ones((length, )), np.zeros((text_len_var-length, ))],
                axis=0
            )
        else:
            sentence_f = np.array(sentence_f, dtype='float32')
            text_data = sentence_f[:text_len_var]
            text_mask = np.ones((text_len_var, ))
        return text_data, text_mask

    def encodeVideo(self, video_name):
        video_data = np.load(os.path.join(self.video_dir, video_name+'.npy'))
        length = video_data.shape[0]
        # if length == 0:
        #     fp.write(video_name+'\n')
        if length >= config.video_frames:
            video_data = video_data[:self.video_len]
            video_mask = np.ones((self.video_len, ))
        else:
            video_data = np.concatenate(
                [
                    video_data,
                    np.zeros(
                        (
                            self.video_len-length,
                            self.video_feature_dim
                        )
                    )
                ], axis=0)
            video_mask = np.concatenate(
                [np.ones((length, )), np.zeros((self.video_len-length, ))],
                axis=0
            )
        return video_data, video_mask

    def selectList(self, data, idxies):
        return [data[idx] for idx in idxies]

    def iterator(self, dataset, shuffle):
        # dataset = np.array(dataset)
        length = len(dataset)
        indice = np.arange(length)
        if shuffle:
            np.random.shuffle(indice)
        for start_idx in range(0, length, self.batch_size):
            if shuffle:
                min_num = min(start_idx+self.batch_size, length)
                excerpt = indice[start_idx:min_num]
            else:
                min_num = min(start_idx+self.batch_size, length)
                excerpt = indice[start_idx:min_num]
            yield self.selectList(dataset, excerpt)

    def loadOneBatch(self, batch):
        num_samples = len(batch)
        videos = []
        video_masks = []
        befores = []
        before_masks = []
        afters = []
        afters_masks = []
        lables = []
        for data in batch:
            video_name = data[0]
            video_f, video_mask = self.encodeVideo(video_name)
            videos.append(video_f)
            video_masks.append(video_mask)
            text_before = data[1]
            before, before_mask = self.encodeSentence(text_before, texttype='before')
            befores.append(before)
            before_masks.append(before_mask)

            text_after = data[2]
            after, after_mask = self.encodeSentence(text_after, texttype='after')
            afters.append(after)
            afters_masks.append(after_mask)

            answer = data[3]
            lables.append(answer)

        videos = np.array(videos, dtype='float32')
        video_masks = np.array(video_masks, dtype='uint8')
        befores = np.array(befores, dtype='float32')
        before_masks = np.array(before_masks, dtype='uint8')
        afters = np.array(afters, dtype='float32')
        afters_masks = np.array(afters_masks, dtype='uint8')
        lables = np.array(lables, dtype='uint8')

        return num_samples, videos, video_masks,\
            befores, before_masks, afters, afters_masks,\
            lables
