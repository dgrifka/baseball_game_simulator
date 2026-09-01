"""Surname extraction for chart labels.

The spray chart labels each notable batted ball with the hitter's surname.
It used to take ``full_name.split()[-1]``, which labels Vladimir Guerrero Jr.
as "Jr." and Elly De La Cruz as "Cruz".
"""

import unittest

from Simulator.visualizations import surname


class TestSurname(unittest.TestCase):

    def test_drops_generational_suffix(self):
        self.assertEqual(surname("Vladimir Guerrero Jr."), "Guerrero")

    def test_keeps_multi_token_surname(self):
        self.assertEqual(surname("Elly De La Cruz"), "De La Cruz")

    def test_drops_suffix_from_two_token_surname_case(self):
        self.assertEqual(surname("Jazz Chisholm Jr."), "Chisholm")

    def test_hyphenated_given_name(self):
        self.assertEqual(surname("Ha-Seong Kim"), "Kim")

    def test_plain_two_token_name(self):
        self.assertEqual(surname("Cal Raleigh"), "Raleigh")

    def test_single_token_name_returns_itself(self):
        self.assertEqual(surname("Ichiro"), "Ichiro")

    def test_empty_string(self):
        self.assertEqual(surname(""), "")

    def test_roman_numeral_suffix(self):
        self.assertEqual(surname("Ken Griffey III"), "Griffey")


if __name__ == "__main__":
    unittest.main()
