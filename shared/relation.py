from typing import List, Tuple


class Relation:
    def __init__(self, index: int, name: str, edges: List[Tuple[int, int]], original_id: str = None):
        """
        Creates a relation object.
        :param original_id: original id in the dataset, can be none.
        :param index: unique index of the relation.
        :param name: the name of the relation, such as written_by, used as book.written_by.author.
        :param edges: list of entity pairs (src, dst), where this relation type appears.
        """
        self.original_id = original_id
        self.index = index
        self.name = name
        self.edges = edges
