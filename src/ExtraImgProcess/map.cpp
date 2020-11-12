#include "map.h"

namespace dso
{

void Map::addKeyFrame(FrameHessian* recordFrame)
{
    recordFrame->isRecord=true;
    if(myMap.currentRecKeyFrameVec.size()==0){
        recordFrame->recordID = 0;
        recordFrame->groupID =myMap.maxGroupID+1;
        myMap.maxGroupID = recordFrame->groupID;
    }
    else{
        FrameHessian* lastFrame = myMap.newComeInFrame;
        recordFrame->recordID=lastFrame->recordID+1;
        recordFrame->groupID = lastFrame->groupID;
        recordFrame->maxGroupID = lastFrame->maxGroupID;
        addConnect(lastFrame,recordFrame);
    }
    myMap.currentRecKeyFrameVec.push_back(recordFrame);
    myMap.newComeInFrame = recordFrame;
}

void Map::addConnect(dso::FrameHessian* Frame1,dso::FrameHessian* Frame2)
{
    connId last(Frame1->groupID,Frame1->recordID),next(Frame2->groupID,Frame2->recordID);
    Frame1->connectID.push_back(next);
    Frame2->connectID.push_back(last);
}
void Map::delConnect(dso::FrameHessian* Frame1,dso::FrameHessian* Frame2)
{
    for(vConnId::iterator iter=Frame1->connectID.begin();iter!=Frame1->connectID.end();iter++)
    {
        if((*iter).first==Frame2->groupID && (*iter).second==Frame2->recordID)
        {
            Frame1->connectID.erase(iter);
            break;
        }
    }
    for(vConnId::iterator iter=Frame2->connectID.begin();iter!=Frame2->connectID.end();iter++)
    {
        if((*iter).first==Frame1->groupID && (*iter).second==Frame1->recordID)
        {
            Frame2->connectID.erase(iter);
            break;
        }
    }
}
//always keep smaller group id
// -1 represent curRecFrames,others are loaded Frames
void Map::updateConnect(std::vector<int> updateList,int minGroupId)
{
    //update currentRecKeyFrameVec
    for(dso::FrameHessian* updateFrame:myMap.currentRecKeyFrameVec)
    {
        connId oringinConnId(updateFrame->groupID,updateFrame->recordID);
        updateFrame->groupID =  minGroupId;
        updateFrame->recordID = myMap.loadRecKeyFrameVVec[minGroupId].size();
        for(connId updateFrameConnId:updateFrame->connectID)
        {
            dso::FrameHessian* connFrame;
            if(updateFrameConnId.first != oringinConnId.first)connFrame = myMap.loadRecKeyFrameVVec[updateFrameConnId.first][updateFrameConnId.second];
            else connFrame = myMap.currentRecKeyFrameVec[updateFrameConnId.second];
            for(int i =0 ;i<connFrame->connectID.size();++i){
                if(connFrame->connectID[i] == oringinConnId){
                    connFrame->connectID[i].first =  updateFrame->groupID;
                    connFrame->connectID[i].second =  updateFrame->recordID;
                    break;
                }
            }
        }
        myMap.loadRecKeyFrameVVec[minGroupId].push_back(updateFrame);

    }
    myMap.currentRecKeyFrameVec.clear();
    //update loadRecKeyFrameVVec
    for(int updateGroupId:updateList){
        if(updateGroupId==-1)continue;
        for(dso::FrameHessian* updateFrame:myMap.loadRecKeyFrameVVec[updateGroupId])
        {
            connId oringinConnId(updateFrame->groupID,updateFrame->recordID);
            updateFrame->groupID =  minGroupId;
            updateFrame->recordID = myMap.loadRecKeyFrameVVec[minGroupId].size();
            for(connId updateFrameConnId:updateFrame->connectID)
            {
                dso::FrameHessian* connFrame = myMap.loadRecKeyFrameVVec[updateFrameConnId.first][updateFrameConnId.second];
                for(int i =0 ;i<connFrame->connectID.size();++i){
                    if(connFrame->connectID[i] == oringinConnId){
                        connFrame->connectID[i].first =  updateFrame->groupID;
                        connFrame->connectID[i].second =  updateFrame->recordID;
                        break;
                    }
                }
            }
            myMap.loadRecKeyFrameVVec[minGroupId].push_back(updateFrame);

        }
        myMap.loadRecKeyFrameVVec[updateGroupId].clear();
    }
    for(vvFrameHessian::const_iterator it=myMap.loadRecKeyFrameVVec.begin();it!=myMap.loadRecKeyFrameVVec.end();)
    {
        if((*it).size()==0)it=myMap.loadRecKeyFrameVVec.erase(it);    //erase删除元素，就会造成迭代器的失效，所以这里要重新指定一个迭代器。
        else ++it;
    }
    for(int i=0;i<myMap.loadRecKeyFrameVVec.size();++i)
    {
        if(myMap.loadRecKeyFrameVVec[i][0]->groupID>i){
            for(dso::FrameHessian* frame:myMap.loadRecKeyFrameVVec[i])
            {
                frame->groupID = i;
                for(int j=0;i<frame->connectID.size();++j)
                {
                    frame->connectID[j].first = i;
                }

            }

        }
    }

}

void Map::mergeKeyFrameLoopInLoadRecFrames(int groupID,int subLoopID)
{
    FrameHessian* loopFrame=myMap.loadRecKeyFrameVVec[groupID][subLoopID];
    myMap.newComeInFrame->needMergedWithLoadFrame = true;
    addConnect(myMap.newComeInFrame,loopFrame);
    myMap.newComeInFrame = loopFrame;
}

void Map::mergeKeyFrameLoopInCurRecFrames(int LoopID)
{
    FrameHessian* loopFrame = myMap.currentRecKeyFrameVec[LoopID];
    addConnect(myMap.newComeInFrame,loopFrame);
    myMap.newComeInFrame = loopFrame;
}

void Map::mergeKeyFrame()
{
    int loadSize = myMap.loadRecKeyFrameVVec.size();
    int maxID=myMap.currentRecKeyFrameVec[0]->groupID;
    if(loadSize<=0){
        myMap.loadRecKeyFrameVVec.push_back(myMap.currentRecKeyFrameVec);
        myMap.maxGroupID = maxID;
    }
    else
    {
        std::vector<int> updateList;
        updateList.push_back(-1);//-1 represent currentRecKeyFrameVec
        int minId=10000;
        int curRecSize = myMap.currentRecKeyFrameVec.size();
        for(int i=0;i<curRecSize;++i){
            if(myMap.currentRecKeyFrameVec[i]->needMergedWithLoadFrame)
            {
                for(int j=0;j<myMap.currentRecKeyFrameVec[i]->connectID.size();j++)
                {
                    int connGroupId = myMap.currentRecKeyFrameVec[i]->connectID[j].first;
                    if(connGroupId!=myMap.currentRecKeyFrameVec[i]->groupID)
                    {
                        updateList.push_back(connGroupId);
                        if(connGroupId<minId)minId=connGroupId;
                        continue;
                    }
                }
                myMap.currentRecKeyFrameVec[i]->needMergedWithLoadFrame = false;
            }
        }
        updateConnect(updateList,minId);

    }
    loadSize = myMap.loadRecKeyFrameVVec.size();
    myMap.maxGroupID = loadSize-1;
    myMap.frameSize.clear();
    for(int i =0;i<loadSize;++i)
        myMap.frameSize.push_back(myMap.loadRecKeyFrameVVec[i].size());
}

bool Map::findId(std::set<int> visited,connId id,int base)
{
    return visited.find(id.first*base+id.second)!=visited.end();
}

vConnId Map::searchPath(connId startId,connId endId)
{
    std::set<int> visited;
    vConnId path;
    std::map<connId,connId> recordConn;
    int base = 10000;
    std::deque<connId> qConnId;
    qConnId.push_front(startId);
    visited.insert(startId.first*base+startId.second);
    bool findPath = false;
    while(!qConnId.empty()){
        connId id = qConnId.front();
        qConnId.pop_front();
        if(id == endId) {
            findPath=true;
            break;
        }
        vConnId connectID;
        if(id.first == myMap.currentRecKeyFrameVec[0]->groupID) connectID = myMap.currentRecKeyFrameVec[id.second]->connectID;
        else connectID = myMap.loadRecKeyFrameVVec[id.first][id.second]->connectID;

        for(int i = 0;i<connectID.size();++i)
        {
            if(!findId(visited,connectID[i],base))
            {
                qConnId.push_front(connectID[i]);
                visited.insert(connectID[i].first*base+connectID[i].second);
                recordConn[connectID[i]]=id;
            }
        }
    }
    if(findPath){
        connId id=endId;
        while(id!=startId){
            path.push_back(id);
            id=recordConn[id];
        }
        path.push_back(id);
        std::reverse(path.begin(),path.end());
    }
    return path;


}

void Map::SaveMap()
{
    mergeKeyFrame();
    std::ofstream out(mapfile, std::ios_base::binary);
    if (!out)
    {
        std::cerr << "Cannot Write to Mapfile: " << mapfile << std::endl;
        exit(-1);
    }
    std::cout << "Saving Mapfile: " << mapfile << std::flush;
    boost::archive::binary_oarchive oa(out, boost::archive::no_header);
    oa << myMap;
    //    oa << mpKeyFrameDatabase;
    std::cout << " ...done" << std::endl;
    out.close();
}
bool Map::LoadMap()
{
    std::ifstream in(mapfile, std::ios_base::binary);
    if (!in)
    {
        std::cerr << "Cannot Open Mapfile: " << mapfile << " , Create a new one" << std::endl;
        return false;
    }
    std::cout << "Loading Mapfile: " << mapfile << std::flush;
    boost::archive::binary_iarchive ia(in, boost::archive::no_header);
    ia >> myMap;

    std::cout << " ...done" << std::endl;
    std::cout << "Map Reconstructing" << std::flush;

    std::cout << " ...done" << std::endl;
    in.close();
    return true;
}

void Map::setDir(std::string dir)
{
    mapfile=dir;
}

void Map::mapTest()
{
    //test merge
    //test findpath
}

template<class Archive>
void MapPool::serialize(Archive &ar, const unsigned int version)
{
    // don't save mutex
    ar & MapPool::loadRecKeyFrameVVec;
    ar & MapPool::maxGroupID;
    ar & MapPool::frameSize;
    ar & MapPool::K;
}
template void MapPool::serialize(boost::archive::binary_iarchive&, const unsigned int);
template void MapPool::serialize(boost::archive::binary_oarchive&, const unsigned int);
}
