# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys

def parse_args(parser, args_list):
    if args_list is None:
        args_list = sys.argv[1:]
    if parse_args is None:
        raise ValueError("Argument parser cannot be None")
    return parser.parse_args(args_list)

def parse_bool(s):
    lower = s.lower()
    if lower in {"1", "true", "y", "yes"}:
        return True
    elif lower in {"0", "false", "n", "no"}:
        return False
    else:
        raise ValueError("Expected boolean value but got '%s'" % (s,))
