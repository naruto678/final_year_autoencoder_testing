import java.util.*;
import java.io.*; 
import java.util.concurrent.*;
import java.util.logging.Logger; 


class Task{

	private final String products[]={"AAPL", "AAL", "GOOGL"}; 
	private final Logger logger=Logger.getLogger(Task.class.getName()); 
	public static void main(String[] args) throws Exception {
		Stream<Price> lines=new BuffeeredReader(new InputStreamReader(System.in))
									.lines()
									.map(processLine)
								   .filter(price ->price!=null); 	
		TradingAlgorithm algo=new AutomatedTrader(products);
		lines.forEach((price)->{
			Trade trade=algo.buildTrades(price);
			System.out.println(trade); 
		});
		

	}	
	private Price processLine(String str){
		String vals[]=str.split(","); 
		try{
			String product_name=vals[0]; 
			double price=Double.parseDouble(vals[1]); 
			return new Price(product_name, price); 
		}catch(Exception e){
			logger.severe("Error occured while parsing "+e);
			return null ;
		}
	}
}


class MarketDataProvider{

}

class Bank{


}


interface TradingAlgorithm{
	Trade buildTrades(Price price); 
}



abstract class GenericAutomatedTrader implements TradingAlgorithm{

	private Map<String,ArrayDeque> map; 
	private final int CAPACITY=4; 
	private Map<String, Semaphore> semaphoreMap; 
	AutomatedTrader(String products[]){
		 this.map=new ConcurrentHashMap<>();
		 
		 for(String product: products){
			 map.computeIfAbsent(product,()->new ArrayDeque<>()); 
			 semaphoreMap.computeIfAbsent(product,()->new Semaphore(1));
		 }

	}


	abstract protected  Trade buildTrade(Price price, ArrayDeque deque) throws Exception; 

	@Override
	public Trade buildTrades(Price price){
		
		
		Optional<ArrayDeque> optionalPriceList=Optional.of(price.getProductName()) // get the productName
												.map((name)->this.map.get(name)); 	// get the corresponding SortedSet for the product
		optionalPriceList.orElseThrow(new RuntimeException("Price list does not exist for the following product "+price)); 
		

		ArrayDeque<Double> priceList=optionalPriceList.get();

		Semaphore semaphore=this.semaphoreMap.get(price.getProductName()); 
		
		semaphore.acquireUninterruptibly(); 
		try{
			buildTrade(price, priceList); 
		}catch(Exception e){
			e.printStackTrace(): 
		}finally{
			semaphore.release(); 
		}

		

						
	}

}

/*
 *this class implements the simple moving window average
 * */
class  AutomatedTrader extends GenericAutomatedTrader{
	private Map<String, Double> prevValue; 

	AutomatedTrader(String products[]){
		super(products);
	   	this.prevValue=new ConcurrentHashMap<String, Double>(); 	
	}

	@Override
	private Trade buildTrade(Price price, ArrayDeque<Double> priceList) throws Exception{
		priceList.add(price.getPrice()); 
		
		if(priceList.size()< CAPACITY) 	
			return null; 
		else if(priceList.size()==CAPACITY){
			double sum=0d; 
			for(double d: priceList)
				sum+=d; 
			map.put(price.getProductName(), sum); 
			if(sum > priceList.peekFirst()*CAPACITY)
				return new Trade(price.getProductName(), Direction.SELL,price.getPrice(),1000d);
		   	else 
				return null ; 	
		}
		else{
			double firstValue=priceList.pollFirst(); 
			double currentValue=map.get(price.getProductName())-firstValue;
		  	if(currentValue > priceList.peekFirst() * CAPACITY){
				map.put(price.getProductName(), currentValue); 
				return new Trade(price.getProductName(), Direction.SELL, price.getPrice(), 1000d);
			}	
			else 
				return null; 
		}
		
	}
}



class Price{
	private String productName; 
	private double price; 
	Price(String productName, double price){
		this.produceName=productName; 
		this.price=price; 
	}
	public String getProductName(){
		return this.productName; 
	}
	public double getPrice(){
		return this.price; 
	}
	@Override
	public String toString(){
		return String.format("Price(%s, %f)", productName, price); 
	}

}


class Trade{
	private String productName; 
	private Direction direction; 
	private double price; 
	private	double quantity; 


	Trade(String productName, Direction direction, double price , double quantity){
		this.productName=productName; 
		this.price=price; 
		this.direction=direction ; 
		this.quanity=quantity; 

	}


	@Override
	public String toString(){
		return String.format("Trade(%s, %s, %d,%d )", produceName, direction, price, quantity); 
	}
}


enum Direction{
	BUY, SELL; 
	
}




