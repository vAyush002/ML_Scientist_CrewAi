from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task

# 🔥 FINAL, CORRECT LLM CONFIG
llm = LLM(
    model="ollama/llama3",   # ✅ PROVIDER + MODEL
    temperature=0.1
)

@CrewBase
class MlScientistCrew:
    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    @agent
    def data_analyst(self) -> Agent:
        return Agent(config=self.agents_config["data_analyst"], llm=llm)

    @agent
    def research_planner(self) -> Agent:
        return Agent(config=self.agents_config["research_planner"], llm=llm)

    @agent
    def model_experimenter(self) -> Agent:
        return Agent(config=self.agents_config["model_experimenter"], llm=llm)

    @agent
    def evaluator(self) -> Agent:
        return Agent(config=self.agents_config["evaluator"], llm=llm)

    @agent
    def scientific_communicator(self) -> Agent:
        return Agent(config=self.agents_config["scientific_communicator"], llm=llm)

    @task
    def dataset_analysis(self) -> Task:
        return Task(config=self.tasks_config["dataset_analysis"], agent=self.data_analyst())

    @task
    def research_planning(self) -> Task:
        return Task(
            config=self.tasks_config["research_planning"],
            agent=self.research_planner(),
            context=[self.dataset_analysis()]
        )

    @task
    def model_training(self) -> Task:
        return Task(
            config=self.tasks_config["model_training"],
            agent=self.model_experimenter(),
            context=[self.dataset_analysis(), self.research_planning()]
        )

    @task
    def model_evaluation(self) -> Task:
        return Task(
            config=self.tasks_config["model_evaluation"],
            agent=self.evaluator(),
            context=[
                self.dataset_analysis(),
                self.research_planning(),
                self.model_training()
            ]
        )

    @task
    def result_explanation(self) -> Task:
        return Task(
            config=self.tasks_config["result_explanation"],
            agent=self.scientific_communicator(),
            context=[
                self.dataset_analysis(),
                self.research_planning(),
                self.model_training(),
                self.model_evaluation()
            ],
            output_file="final_report1.md"
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            llm=llm,
            verbose=True
        )
