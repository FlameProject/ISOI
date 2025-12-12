# В simple_tester.py исправьте функцию recognize_symbol:

def recognize_symbol(self, tensor_img):
    """Распознать символ"""
    try:
        with torch.no_grad():
            tensor_img = tensor_img.to(self.device)
            output = self.model(tensor_img)
            probabilities = F.softmax(output, dim=1)
            
            # Топ-5 вариантов
            top5_probs, top5_indices = torch.topk(probabilities, 5)
            
            results = []
            for i in range(min(5, len(top5_indices[0]))):
                idx = top5_indices[0][i].item()
                prob = top5_probs[0][i].item()
                
                if idx < len(self.config.chars):
                    char = self.config.chars[idx]
                    results.append(f"'{char}': {prob:.2%}")
                else:
                    results.append(f"[индекс {idx} вне диапазона]: {prob:.2%}")
            
            return f"Ожидаемый: '{self.current_char}'\n" + "\n".join(results)
            
    except Exception as e:
        return f"Ошибка распознавания: {e}"