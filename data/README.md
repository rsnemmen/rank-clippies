Some notes about the rankings in the data files
=========================================

One has to be careful with the data provided by  https://artificialanalysis.ai/. When you open the main website, the plots show only a selection of LLM models. 

The Terminal Bench 2.0 scores quoted by many vendors (mostly the Chinese vendors such as ) differ from the ones available in the public website https://www.tbench.ai/leaderboard/terminal-bench/2.0.

The Wolfram notebook `rank_convert.nb` uniformizes and converts from such scores to a ranking, taking into account how many models were evaluated in each benchmark. 
