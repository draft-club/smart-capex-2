import requests
requests.get('http://10.238.36.21:5001/',
             json={'path':'/mnt/sample.csv',
                   'file_type' : 'oss',
                   'delimiter':',',
                   'init':False
                  })

requests.get('http://10.238.36.21:5001/',
             json={'path':'',
                   'file_type' : 'cdr',
                   'delimiter':',',
                   'init':False,
                   'join_paths': {'cp': {'path': '', 'delimiter': ''}}
                  })

requests.get('http://10.238.36.21:5001/',
             json={'path':'',
                   'file_type' : 'cp',
                   'delimiter':',',
                   'init':False
                  })

requests.get('http://10.238.36.21:5001/',
             json={'path':'',
                   'file_type' : 'cells',
                   'delimiter':',',
                   'init':False,
                   'join_paths': {'cp': {'path': '', 'delimiter': ''}}
                  })
requests.get('http://10.238.36.21:5001/',
             json={'path':'',
                   'file_type' : 'sites',
                   'delimiter':',',
                   'init':False,
                   'join_paths': {'sites': {'path': '', 'delimiter': ''}}
                  })