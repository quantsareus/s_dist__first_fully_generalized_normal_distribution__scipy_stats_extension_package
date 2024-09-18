#! /bin/python3



#####################################################################################

##### For testing build: python3 setup.py build

##### For install:
##### cd ~/anaconda3/
##### bin/pip3 install .../s_dist


#####################################################################################



from distutils.core import setup

setup(
	name="s_dist",
	version= "0.32",
	author= "Schlingmann, H",
	author_email= "email{at]quantsareus[dot}net",
	maintainer= "QuantsAreUs",
	maintainer_email= "email{at]quantsareus[dot}net",
	url= "github.com/quantsareus/s_dist__first_fully_generalized_normal_distribution__scipy_stats_extension_package",
	download_url= "github.com/quantsareus/s_dist__first_fully_generalized_normal_distribution__scipy_stats_extension_package/",
	license= "GPLv3",
	# packages= subdirectories to be installed containing a __init__.py ; should not have the same name as distribution:
	packages= ["s_dist", "s_dist/casestudies", "s_dist/dsets", "s_dist/test", "s_dist/utils"],
	package_data= {"s_dist": ["table.csv"], "s_dist/dsets": ["*.csv"] },
	# further data files to be included in the distribution archive but not to get installed: 
	# data_files= [(".", ["MANIFEST"])],
	# python modules in the project's root directory to be installed without directory and to get imported into the global namespace:
	# py_modules= ["global_modul"],
	# scripts in the project's root directory not to be included in the distribution archive (e.g. building scripts):
	# scripts= [""],
	description= "A mainly uninteresting number crunching modul",
	long_description="From 100 _random_ people discovering the package, most probably about 97 (or even more) will find it uninteresting. It is a special hobby"
	)

