#!/bin/bash

# let there be darkness



for value in {165..380}
do
  java -jar ManKalah.jar "java -jar test_agents/JimmyPlayer.jar" "python3 magent/observer_agent.py -r ${value} -c north-jimmy-"&
  java -jar ManKalah.jar "java -jar test_agents/error404.jar" "python3 magent/observer_agent.py -r ${value} -c north-error404-"&
 # java -jar ManKalah.jar "java -jar test_agents/Group2Agent.jar" "python3 magent/observer_agent.py -r ${value} -c north-group2-"&

  java -jar ManKalah.jar "python3 magent/observer_agent.py -r ${value} -c south-jimmy-" "java -jar test_agents/JimmyPlayer.jar"  &
  java -jar ManKalah.jar "python3 magent/observer_agent.py -r ${value} -c south-error404-" "java -jar test_agents/error404.jar"  &
 # java -jar ManKalah.jar "python3 magent/observer_agent.py -r ${value} -c south-group2-"  "java -jar test_agents/Group2Agent.jar" &

  wait
done
# may the hack be with you

