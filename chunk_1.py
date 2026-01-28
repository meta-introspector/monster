    def test_typo(self): #771
        L = self.tc.get('/ModularForm/GL2/TotallyReal/?field_label=2.2.5.1')
        assert 'Search again' in L.get_data(as_text=True)