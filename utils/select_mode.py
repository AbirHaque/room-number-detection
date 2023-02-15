#This Function is used for adding new Keypoints
class select_mode:
    def select_mode(key, mode):

        number = -1
        if 48 <= key <= 57:  # 0 ~ 9
            number = key - 48
        if key == 110:  # n
            mode = 0
        if key == 107:  # k
            mode = 1
        return number, mode