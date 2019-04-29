all: cluonDataStructures_pb2.py message_set_pb2.py

cluonDataStructures.proto: ../message-set/cluonDataStructures.odvd
	cluon-msc --proto --out=cluonDataStructures.proto ../message-set/cluonDataStructures.odvd

cluonDataStructures_pb2.py: cluonDataStructures.proto
	protoc --python_out=. cluonDataStructures.proto

message_set.proto: ../message-set/message-set.odvd
	cluon-msc --proto --out=message_set.proto ../message-set/message-set.odvd

message_set_pb2.py: message_set.proto
	protoc --python_out=. message_set.proto

clean:
	rm -f *_pb2.py *pyc *.proto 
