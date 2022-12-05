```
kind create cluster
kind create cluster -n <cluster-name>

kubectl get svc
kubectl get pod
kubectl get deployment

kubectl apply -f deployment.yaml
kubectl apply -f service.yaml

kind load docker-image ping:v001
kubectl port-forward <pod-name> 9696:9696

kubectl apply -f service.yaml
kubectl port-forward service/ping 8080:80

kind load docker-image zoomcamp-10-model:xception-v4-001
kind load docker-image <docker-image> -n <cluster-name> 

kubectl exec -it <pod-name> -- bash

apt update
apt install curl telnet
telnet tf-serving-clothing-model.default.svc.cluster.local 8500

eksctl create cluster --name <cluster-name> -f <config-file>

aws ecr create-repository --repository-name <repo-name>
aws ecr get-login --no-include-email
docker push ${model-tag}

eksctl delete cluster --name <cluster-name>
```