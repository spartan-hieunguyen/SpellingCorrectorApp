class VietnameseNormalizer:
    def __init__(self) -> None:
        self.bang_nguyen_am = [['a', 'à', 'á', 'ả', 'ã', 'ạ'],
                        ['ă', 'ằ', 'ắ', 'ẳ', 'ẵ', 'ặ'],
                        ['â', 'ầ', 'ấ', 'ẩ', 'ẫ', 'ậ'],
                        ['e', 'è', 'é', 'ẻ', 'ẽ', 'ẹ'],
                        ['ê', 'ề', 'ế', 'ể', 'ễ', 'ệ'],
                        ['i', 'ì', 'í', 'ỉ', 'ĩ', 'ị'],
                        ['o', 'ò', 'ó', 'ỏ', 'õ', 'ọ'],
                        ['ô', 'ồ', 'ố', 'ổ', 'ỗ', 'ộ'],
                        ['ơ', 'ờ', 'ớ', 'ở', 'ỡ', 'ợ'],
                        ['u', 'ù', 'ú', 'ủ', 'ũ', 'ụ'],
                        ['ư', 'ừ', 'ứ', 'ử', 'ữ', 'ự'],
                        ['y', 'ỳ', 'ý', 'ỷ', 'ỹ', 'ỵ']]
        self.nguyen_am_to_ids = {}
        self._initialize_vowels()
    
    def _initialize_vowels(self):
        for i in range(len(self.bang_nguyen_am)):
            for j in range(len(self.bang_nguyen_am[i])):
                self.nguyen_am_to_ids[self.bang_nguyen_am[i][j]] = (i, j)

    def _is_valid_vietnam_word(self, word):
        chars = list(word)
        nguyen_am_index = -1
        for index, char in enumerate(chars):
            x, _ = self.nguyen_am_to_ids.get(char, (-1, -1))
            if x != -1:
                if nguyen_am_index == -1:
                    nguyen_am_index = index
                else:
                    if index - nguyen_am_index != 1:
                        return False
                    nguyen_am_index = index
        return True

    def normalize(self, word):
        if not self._is_valid_vietnam_word(word):
            return word

        chars = list(word)
        dau_cau = 0
        nguyen_am_index = []
        qu_or_gi = False
        for index, char in enumerate(chars):
            x, y = self.nguyen_am_to_ids.get(char, (-1, -1))
            if x == -1:
                continue
            elif x == 9:  # check qu
                if index != 0 and chars[index - 1] == 'q':
                    chars[index] = 'u'
                    qu_or_gi = True
            elif x == 5:  # check gi
                if index != 0 and chars[index - 1] == 'g':
                    chars[index] = 'i'
                    qu_or_gi = True
            if y != 0:
                dau_cau = y
                chars[index] = self.bang_nguyen_am[x][0]
            if not qu_or_gi or index != 1:
                nguyen_am_index.append(index)
        if len(nguyen_am_index) < 2:
            if qu_or_gi:
                if len(chars) == 2:
                    x, y = self.nguyen_am_to_ids.get(chars[1])
                    chars[1] = self.bang_nguyen_am[x][dau_cau]
                else:
                    x, y = self.nguyen_am_to_ids.get(chars[2], (-1, -1))
                    if x != -1:
                        chars[2] = self.bang_nguyen_am[x][dau_cau]
                    else:
                        chars[1] = self.bang_nguyen_am[5][dau_cau] if chars[1] == 'i' else self.bang_nguyen_am[9][dau_cau]
                return ''.join(chars)
            return word

        for index in nguyen_am_index:
            x, y = self.nguyen_am_to_ids[chars[index]]
            if x == 4 or x == 8:  # ê, ơ
                chars[index] = self.bang_nguyen_am[x][dau_cau]
                return ''.join(chars)

        if len(nguyen_am_index) == 2:
            if nguyen_am_index[-1] == len(chars) - 1:
                x, y = self.nguyen_am_to_ids[chars[nguyen_am_index[0]]]
                chars[nguyen_am_index[0]] = self.bang_nguyen_am[x][dau_cau]
            else:
                x, y = self.nguyen_am_to_ids[chars[nguyen_am_index[1]]]
                chars[nguyen_am_index[1]] = self.bang_nguyen_am[x][dau_cau]
        else:
            x, y = self.nguyen_am_to_ids[chars[nguyen_am_index[1]]]
            chars[nguyen_am_index[1]] = self.bang_nguyen_am[x][dau_cau]
        return ''.join(chars)