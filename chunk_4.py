    def display(self, rec):
        ans = self.get(rec)
        if ans > 0:
            return "&#x2713;"
        elif ans < 0:
            return self.no
        else:
            return self.unknown