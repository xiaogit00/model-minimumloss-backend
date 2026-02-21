from services import file_db
import unittest
import asyncio
from pathlib import Path
class TestDB(unittest.TestCase):
    def test_get_models(self):
        res = file_db.get_models()
        print(res)

if __name__ == '__main__':
    asyncio.run(unittest.main())

