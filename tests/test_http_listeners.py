import os
import sys
import unittest

parent_dir = os.path.abspath(os.path.join(__file__, "..", ".."))
sys.path.append(parent_dir)

import koboldcpp


class ListenerThreadSocketTests(unittest.TestCase):
    def test_each_available_listener_gets_a_full_worker_pool(self):
        cases = (
            (True, True, 24, 24),
            (True, False, 24, 0),
            (False, True, 0, 24),
            (False, False, 0, 0),
        )

        for ipv4_available, ipv6_available, expected_ipv4, expected_ipv6 in cases:
            with self.subTest(ipv4=ipv4_available, ipv6=ipv6_available):
                ipv4_sock = object() if ipv4_available else None
                ipv6_sock = object() if ipv6_available else None

                thread_sockets = koboldcpp._get_listener_thread_sockets(
                    ipv4_sock, ipv6_sock, 24
                )

                self.assertEqual(
                    sum(server_socket is ipv4_sock for server_socket in thread_sockets),
                    expected_ipv4,
                )
                self.assertEqual(
                    sum(server_socket is ipv6_sock for server_socket in thread_sockets),
                    expected_ipv6,
                )


if __name__ == "__main__":
    unittest.main()
