"""
Investment Advisor Agent — Internal Tool Modules (智能投顾内部工具模块)

Single master agent + tool modules architecture.
All modules are rule-based for MVP; no LLM calls, no real vector DB.

Main pipeline:
  ProfileAnalyzer → GoalPlanner → RiskAssessor → AllocationEngine → AdvisoryCompliancePolicy
"""
