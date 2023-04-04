%----------------------------------------------------------------------------%
%  Binary Anarchich Society Algorithm (BASO) source codes                    %
%  for Feature Selection                                                     %
%                                                                            %
%  Umit Kilic                                                                %
%                                                                            %
%  email: ukilic@atu.edu.tr & umitkilic21@gmail.com                       %
%----------------------------------------------------------------------------%

function [sF,sFNo,sFidx,curve]=bASO(F,L,N,T)
%-------------------- INPUT -----------------------
% F: feature vector
% L: Labels
% N: number of population
% T: max num of iteration
%
%-------------------- OUTPUT -----------------------
% sF:       selected features
% sFNo:     Number of selected features
% sFidx:    selected features index
% curve:    convergence curve
%
%---------------------------------------------------

% objective function
global fun;
fun=@FitnessFunction;

% number of dimensions
global D;
D=size(F,2);
D
%% population struct

Member.Position=zeros(1,D);
Member.ErrorRate=[];                % error rate
Member.Accuracy=[];                 % 1 - error rate
Member.Fmeasure=[];                 % micro averaged f-measure
Member.fi=[];                       % Ficklenes Index
Member.ei=[];                       % External Irregulariy Index
Member.ii=[];                       % Internal Irregularity Index
Member.Best.Position=zeros(1,D);    % Personal Best Position
Member.Best.Fmeasure=[];            % Personel Best F-measure

global GB;
GB.Position=zeros(1,D);             % Global Best Position
GB.Fmeasure=-inf;                   % Global Best F-measure

global IterBestIdx;                     % Index of the Iteration Best

%create population array
global members;
members=repmat(Member,N,1);

%% Initialize the population
for i=1:N
    for d=1:D
       if rand() > 0.5
            members(i).Position(1,d)=1;
       end
    end
    
    %in case of having zero for all vector, select a position and turn it
    %to 1.
    if sum(members(i).Position)==0
       members(i).Position(1,randi([1,D],1))=1; 
    end
end

% calculate measures for initialized members
for i=1:N
    [members(i).ErrorRate,members(i).Fmeasure,members(i).Accuracy]=fun(F,L,members(i).Position(1,:));
    members(i).Best.Position=members(i).Position; members(i).Best.Fmeasure=members(i).Fmeasure;
end

global idx;

[~,idx]=sort([members(:).Fmeasure],'descend');

IterBestIdx=idx(1);
GB.Position=members(idx(1)).Position; GB.Fmeasure=members(idx(1)).Fmeasure;

curve=inf; t=1;
%figure(1); clf; axis([1 50 0 0.5]); xlabel('Number of Iterations');
%ylabel('Fitness Value'); title('Convergence Curve'); grid on;
%% Main Loop of BASO

while t<=T
global members;


    % calculate Fickleness Index, External Irregularity Index and Internal Irregularity Index
    CalculateFI_EI_II(N);

    %beta1=-0.001; beta2=-0.2; beta3=-0.08;
    
    beta1=mean([members(:).fi]);
    beta2=mean([members(:).ei]);
    beta3=mean([members(:).ii]);

    for i=1:N
        %movement Policy Current (MPi_Current)
        [mpi_current_pos,mpi_current_fmeasure]=MPi_Current(i,N,beta1,F,L);
        
        % Movement Policy Society (MPi_Society)
        [mpi_society_pos,mpi_society_fmeasure]=MPi_Soceity(i,N,beta2,F,L);
        
        % Movement Policy Past (MPi_past)
        [mpi_past_pos,mpi_past_fmeasure]=MPi_Past(i,N,beta3,F,L);
        
        [~,maxid]=max([mpi_current_fmeasure,mpi_society_fmeasure,mpi_past_fmeasure]);
%         disp(['Values and Pos: Curr:' num2str(mpi_current_fmeasure) ' Pos: ' num2str(mpi_current_pos)...
%             ' Soc: ' num2str(mpi_society_fmeasure) ' Pos: ' num2str(mpi_society_pos)...
%             ' Past: ' num2str(mpi_past_fmeasure) ' Pos: ' num2str(mpi_past_pos) ] );

        % elitism movement policy
        if maxid==1
            members(i).Position=mpi_current_pos;
        elseif maxid==2
            members(i).Position=mpi_society_pos;
        elseif maxid==3
            members(i).Position=mpi_past_pos;
        end
    end
       
    for i=1:N
        [members(i).ErrorRate,members(i).Fmeasure,members(i).Accuracy]=fun(F,L,members(i).Position(1,:));
        if members(i).Best.Fmeasure<members(i).Fmeasure
            members(i).Best.Position=members(i).Position; members(i).Best.Fmeasure=members(i).Fmeasure;
        end
        
    end
    
    [~,idx]=sort([members(:).Fmeasure],'descend');
    
    IterBestIdx=idx(1);
    if members(idx(1)).Fmeasure > GB.Fmeasure
        GB.Position=members(idx(1)).Position; GB.Fmeasure=members(idx(1)).Fmeasure;
    end
    
    disp(['Iteration: ' num2str(t) ' Iteration Best: ' num2str(members(IterBestIdx).Fmeasure)...
        ' GlobalBest: ' num2str(GB.Fmeasure)]);
    
    curve(t)=GB.Fmeasure;
    % Plot convergence curve
    %pause(0.000000001); hold on;
    %CG=plot(t,GB.Fmeasure(1,1),'Color','r','Marker','.'); set(CG,'MarkerSize',5);
    t=t+1;
end

% select features based on selected index
Pos=1:D; sF=Pos(GB.Position==1); sFNo=length(sF); sFidx=F(:,sF);


end

function CalculateFI_EI_II(N)
global members;
global IterBestIdx;
global idx;
global GB;

% calculate Fickleness Index, External Irregularity Index and Internal Irregularity Index
    for i=1:N
        % calculate fickleness irregularity index
        members(i).fi=(members(i).Fmeasure - members(IterBestIdx).Fmeasure)/members(IterBestIdx).Fmeasure;

        % calculate external irregularity index 
        if i==idx(N)
            members(i).ei=((members(idx(N-1)).Fmeasure - members(IterBestIdx).Fmeasure)/members(IterBestIdx).Fmeasure);
        else 
            members(i).ei=((members(idx(N)).Fmeasure - members(IterBestIdx).Fmeasure)/members(IterBestIdx).Fmeasure);
        end

        % calculate internal irregularity index
        members(i).ii=((members(i).Best.Fmeasure - GB.Fmeasure)/GB.Fmeasure);
    end
end

function [pos,fm]=MPi_Current(i,N,beta1,F,L)
global IterBestIdx;
global members;
global GB;
global fun;
global D;

    %movement Policy Current (MPi_Current)
    
       if i~=IterBestIdx
           if members(i).fi <= beta1
               % create a random member. If there is no 1 in the member's positions, then add 1 to a random position
                random_mem=randi([0,1],[1,D]); while(sum(random_mem)==0) random_mem=randi([0,1],[1,D]); end
                pos = SPCrossOver(members(i).Position,random_mem);
                
                if sum(pos)==0 pos(1,randi([1,D],1))=1; end
           else
               % find a random id. if id is equal to the related member's id, then re-create
               random_idx=randi([1,N],1); while(random_idx==i) random_idx=randi([1,N],1); end
               
               pos = SPCrossOver(members(i).Position,members(random_idx).Position);
               %in case of having zero for all vector, select a position and turn it to 1.
               if sum(pos)==0 pos(1,randi([1,D],1))=1; end
           end

       else
           
            pos = SPCrossOver(members(i).Position,GB.Position);
           %in case of having zero for all vector, select a position and turn it to 1.
            if sum(pos)==0 disp(['The pos is: ' num2str(pos(1,:)) 'sum is zero.']); pos(1,randi([1,D],1))=1; end
       end
       %disp(['pos bef func: ' num2str(pos(1,:))]);
       [~,fm,~]=fun(F,L,pos(1,:));
    
end

function [pos,fm]=MPi_Soceity(i,N,beta2,F,L)
global members;
global GB;
global IterBestIdx;
global fun;
global D;

    
       if members(i).ei <= beta2
            pos = SPCrossOver(members(i).Position,members(IterBestIdx).Position);
            %in case of having zero for all vector, select a position and turn it to 1.
            if sum(pos)==0 pos(1,randi([1,D],1))=1; end
       else
           % find a random id. if id is equal to the related member's id, then re-create
           random_idx=randi([1,N],1); while(random_idx==i) random_idx=randi([1,N],1); end

           pos = SPCrossOver(members(i).Position,members(random_idx).Position);
           %in case of having zero for all vector, select a position and turn it to 1.
           if sum(pos)==0 pos(1,randi([1,D],1))=1; end
       end
    
       [~,fm,~]=fun(F,L,pos(1,:));
end

function [pos,fm]=MPi_Past(i,N,beta3,F,L)
global members;
global fun;
global D;
    
       if members(i).ii <= beta3
            pos = SPCrossOver(members(i).Position,members(i).Best.Position);
            %in case of having zero for all vector, select a position and turn it to 1.
            if sum(pos)==0 pos(1,randi([1,D],1))=1; end
       else
           % find a random id. if id is equal to the related member's id, then re-create
           random_idx=randi([1,N],1); while(random_idx==i) random_idx=randi([1,N],1); end

           pos = SPCrossOver(members(i).Position,members(random_idx).Position);
           %in case of having zero for all vector, select a position and turn it to 1.
           if sum(pos)==0 pos(1,randi([1,D],1))=1; end
       end
       
       [~,fm,~]=fun(F,L,pos(1,:));
    
end

% single-point crossover (toward Global best)
function [pos]=SPCrossOver(mem1,GBmem)
% mem1:     the member to be changed with crossover
% GBmem:    Global best (used for one point cross over toward it)
global D;
    p=randi([2,D],1);
    
    temp_b=GBmem(p:end);
    GBmem(p:end)=mem1(p:end);
    mem1(p:end)=temp_b;
    pos = mem1;
end

