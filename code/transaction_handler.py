import re
HASHTAG_PATTERN = re.compile(r'#\w*')
MENTION_PATTERN = re.compile(r'@\w*')


class Tweet:
    def __str__(self):
        keyphrases = [word.replace(' ', '_').strip() for word in self.words]
        keyphrases = ' '.join([keyphrase for keyphrase in keyphrases
                               if keyphrase])
        fields = [str(field) for field in self.id, self.uid, self.ts,
                  ' '.join(self.words), ' '.join(self.words_0),
                  ' '.join(self.words_1), ' '.join(self.words_2)]
        return '\x01'.join(fields)+'\n'

    def load_tweet(self, line):
        self.line = line
        items = line.split('\x01')
        # print(items[0])

        self.id = float(items[0])
        self.uid = float(items[1])
        self.ts = int(float(items[2]))
        self.text = items[3]
        self.words = self.text.split(' ')
        self.words_0 = []
        self.words_1 = []
        self.words_2 = []

        if len(items) > 4:
            self.values = [float(k) for k in items[4].split(' ')]
            assert len(self.values) == len(self.words)
        else:
            self.values = [1 for w in self.words]
