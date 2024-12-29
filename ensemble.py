from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression

class DeceptionEnsemble:
    def __init__(self, models):
        self.base_models = models
        
    def create_voting_ensemble(self):
        """创建投票集成模型"""
        estimators = [
            (f'model_{i}', model) 
            for i, model in enumerate(self.base_models)
        ]
        
        return VotingClassifier(
            estimators=estimators,
            voting='soft'
        )
    
    def create_stacking_ensemble(self):
        """创建堆叠集成模型"""
        estimators = [
            (f'model_{i}', model) 
            for i, model in enumerate(self.base_models)
        ]
        
        return StackingClassifier(
            estimators=estimators,
            final_estimator=LogisticRegression(),
            cv=5
        ) 